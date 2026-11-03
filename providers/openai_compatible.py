"""
Pass-through provider for any upstream that already speaks the OpenAI
chat-completions API (llama.cpp's llama-server, vLLM, ...).

The request body is forwarded untouched apart from `model`, so tools,
tool_choice, response_format, stream and every sampling parameter reach the
upstream exactly as the client sent them. The response — JSON or SSE — is
returned untouched too.
"""
from collections.abc import AsyncIterator

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


class UpstreamError(Exception):
    """
    The upstream answered with an error, or could not be reached.

    `status` is the HTTP status this server should answer with, `payload` the
    upstream's error body when it sent one (already OpenAI-shaped for an
    OpenAI-compatible upstream), and `headers` any headers worth forwarding,
    such as Retry-After on a 429.
    """

    def __init__(self, status: int, message: str, payload: dict | None = None, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.payload = payload
        self.headers = headers or {}


class OpenAICompatibleProvider:

    def __init__(self, name: str, base_url: str, api_key: str | None, timeout: float):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # Created on first use, not at import: PROVIDERS is built at import
        # time and provider __init__ must stay cheap.
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _send(self, body: dict, stream: bool) -> httpx.Response:
        client = self._get_client()
        request = client.build_request("POST", "/chat/completions", json=body)
        try:
            response = await client.send(request, stream=stream)
        except httpx.TimeoutException as e:
            logger.error("%s upstream timed out: %s", self.name, e)
            raise UpstreamError(504, f"Upstream '{self.name}' timed out.")
        except httpx.HTTPError as e:
            logger.error("%s upstream unreachable: %s", self.name, e)
            raise UpstreamError(502, f"Upstream '{self.name}' is unreachable at {self.base_url}.")

        if response.status_code >= 400:
            raw = await response.aread()
            await response.aclose()
            raise self._to_error(response, raw)

        return response

    def _to_error(self, response: httpx.Response, raw: bytes) -> UpstreamError:
        try:
            payload = httpx.Response(200, content=raw).json()
            if not isinstance(payload, dict) or "error" not in payload:
                payload = None
        except ValueError:
            payload = None

        message = raw.decode(errors="replace")[:500]
        if payload and isinstance(payload["error"], dict):
            message = payload["error"].get("message", message)

        # A client error is the caller's to fix, so pass its status through.
        # An upstream 5xx is not this server failing, so it becomes a 502.
        status = response.status_code if response.status_code < 500 else 502
        headers = {}
        if "retry-after" in response.headers:
            headers["Retry-After"] = response.headers["retry-after"]

        logger.warning("%s upstream error %s: %s", self.name, response.status_code, message)
        return UpstreamError(status, message, payload, headers)

    async def chat_completion(self, body: dict) -> dict:
        response = await self._send(body, stream=False)
        try:
            return response.json()
        except ValueError:
            raise UpstreamError(502, f"Upstream '{self.name}' returned a non-JSON response.")

    async def open_stream(self, body: dict) -> AsyncIterator[bytes]:
        """
        Start a streaming completion and return an iterator over the raw SSE bytes.

        The upstream request is sent here, before any byte reaches the client,
        so an upstream error still surfaces as a proper HTTP error rather than
        a 200 whose stream breaks off.
        """
        response = await self._send(body, stream=True)

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                # Also runs when the client disconnects mid-stream.
                await response.aclose()

        return relay()
