"""
Dispatch for the OpenAI-compatible /v1/chat/completions endpoint.

The provider is chosen by a prefix on the model name, split on the FIRST slash
only, because upstream model ids contain slashes of their own:

    llamacpp/qwen3-8b                 -> llamacpp, "qwen3-8b"
    llamacpp/Qwen/Qwen3-8B-GGUF       -> llamacpp, "Qwen/Qwen3-8B-GGUF"
    local/minichat-1.5-3b             -> local,    "minichat-1.5-3b"

`llamacpp` is the main path for local models: llama-server gives real
streaming and tool calling. OpenAI-compatible upstreams get the body forwarded untouched. The in-process
`local` provider (transformers) is adapted: messages are rendered through the
model's prompt template, and features it cannot honour are refused with a 400
rather than silently dropped.
"""
import json
import time
import uuid
from collections.abc import AsyncIterator

from langsmith import traceable

from providers.openai_compatible import OpenAICompatibleProvider
from router import PROVIDERS
from settings import OPENAI_COMPATIBLE_UPSTREAMS, UPSTREAM_TIMEOUT_SECONDS
from utils.logger import get_logger

logger = get_logger(__name__)

UPSTREAMS: dict[str, OpenAICompatibleProvider] = {
    name: OpenAICompatibleProvider(name, cfg["base_url"], cfg.get("api_key"), UPSTREAM_TIMEOUT_SECONDS)
    for name, cfg in OPENAI_COMPATIBLE_UPSTREAMS.items()
}

# The same LocalService instance /generate uses, so both endpoints share one
# GPU lock, one admission queue and one resident model.
LOCAL = PROVIDERS["local"]

AVAILABLE_PROVIDERS = sorted([*UPSTREAMS, "local"])


def parse_model(model: str) -> tuple[str, str]:
    provider, sep, name = model.partition("/")
    if not sep or not name:
        raise ValueError(
            f"Model '{model}' has no provider prefix. Use '<provider>/<model>', "
            f"e.g. 'llamacpp/qwen3-8b' or 'local/minichat-1.5-3b'. Providers: {AVAILABLE_PROVIDERS}"
        )
    if provider not in UPSTREAMS and provider != "local":
        raise ValueError(f"Unknown provider: '{provider}'. Available: {AVAILABLE_PROVIDERS}")
    return provider, name


async def close_upstreams() -> None:
    for upstream in UPSTREAMS.values():
        await upstream.aclose()


def _log_usage(provider: str, model: str, response: dict, elapsed: float) -> None:
    usage = response.get("usage") or {}
    logger.info(
        "Chat completion — provider: %s, model: %s, latency: %.3fs, prompt_tokens: %s, completion_tokens: %s",
        provider, model, elapsed, usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )


@traceable(run_type="llm", name="llm-server.chat")
async def chat_completion(body: dict) -> dict:
    provider, model = parse_model(body["model"])
    start = time.perf_counter()

    if provider == "local":
        messages, generation_kwargs = _local_request(body)
        text = await LOCAL.chat(model, messages, **generation_kwargs)
        response = _local_completion(body["model"], text)
    else:
        response = await UPSTREAMS[provider].chat_completion({**body, "model": model})

    _log_usage(provider, model, response, time.perf_counter() - start)
    return response


async def chat_completion_stream(body: dict) -> AsyncIterator[bytes]:
    """
    Returns an iterator of SSE bytes. Any error is raised here, before the
    first byte, so it can still become a proper HTTP error response.
    """
    provider, model = parse_model(body["model"])
    logger.info("Streaming chat completion — provider: %s, model: %s", provider, model)

    if provider == "local":
        # transformers generation here is not incremental, so the whole answer
        # arrives as one chunk. Clients still get a valid stream to consume.
        messages, generation_kwargs = _local_request(body)
        text = await LOCAL.chat(model, messages, **generation_kwargs)
        return _local_stream(body["model"], text)

    return await UPSTREAMS[provider].open_stream({**body, "model": model})


# ---------------------------------------------------------------------------
# Local (transformers) adapter
# ---------------------------------------------------------------------------

_ROLE_ALIASES = {"developer": "system"}


def _text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError("The local provider only accepts text message content.")
            parts.append(part.get("text", ""))
        return "".join(parts)
    raise ValueError("Message content must be a string or a list of text parts.")


def _local_request(body: dict) -> tuple[list[dict], dict]:
    if body.get("tools") or body.get("functions"):
        raise ValueError(
            "The local provider does not support tool calling. "
            "Serve the model through llama-server and use the 'llamacpp/' prefix instead."
        )
    if (body.get("n") or 1) > 1:
        raise ValueError("The local provider only supports n=1.")

    messages = []
    for m in body["messages"]:
        role = _ROLE_ALIASES.get(m.get("role"), m.get("role"))
        if role not in ("system", "user", "assistant") or m.get("tool_calls"):
            raise ValueError(
                f"The local provider cannot handle '{m.get('role')}' messages or tool calls."
            )
        messages.append({"role": role, "content": _text_content(m.get("content"))})

    kwargs = {}
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens is not None:
        kwargs["max_new_tokens"] = int(max_tokens)
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            kwargs[key] = body[key]
    stop = body.get("stop")
    if stop:
        kwargs["stop"] = [stop] if isinstance(stop, str) else list(stop)

    return messages, kwargs


def _local_completion(model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


async def _local_stream(model: str, text: str) -> AsyncIterator[bytes]:
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"
