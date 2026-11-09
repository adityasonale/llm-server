"""
Dispatch for the OpenAI-compatible /v1/chat/completions endpoint.

A plain model name registered in models.yaml goes to the model manager, which
starts the right backend plugin for that model's format and this machine (see
model_manager.py). `llamacpp/<name>` is accepted as an alias for a registered
`<name>`, so clients written against the old prefix keep working.

Otherwise the provider is chosen by a prefix on the model name, split on the
FIRST slash only, because upstream model ids contain slashes of their own:

    llamacpp/qwen3-8b                 -> llamacpp, "qwen3-8b"
    llamacpp/Qwen/Qwen3-8B-GGUF       -> llamacpp, "Qwen/Qwen3-8B-GGUF"

Upstreams are servers someone else started (e.g. llama-server on
LLAMACPP_BASE_URL); the body is forwarded untouched apart from `model`.
"""
import time
from collections.abc import AsyncIterator

from langsmith import traceable

from model_manager import MANAGER
from providers.openai_compatible import OpenAICompatibleProvider
from settings import OPENAI_COMPATIBLE_UPSTREAMS, UPSTREAM_TIMEOUT_SECONDS
from utils.logger import get_logger

logger = get_logger(__name__)

UPSTREAMS: dict[str, OpenAICompatibleProvider] = {
    name: OpenAICompatibleProvider(name, cfg["base_url"], cfg.get("api_key"), UPSTREAM_TIMEOUT_SECONDS)
    for name, cfg in OPENAI_COMPATIBLE_UPSTREAMS.items()
}

AVAILABLE_PROVIDERS = sorted(UPSTREAMS)

# Pseudo-provider for models served by the model manager.
MANAGED = "managed"
# Prefixes that may name a models.yaml entry, for clients that predate it.
_MANAGED_ALIASES = ("llamacpp",)


def parse_model(model: str) -> tuple[str, str]:
    if MANAGER.has(model):
        return MANAGED, model

    provider, sep, name = model.partition("/")
    if provider in _MANAGED_ALIASES and MANAGER.has(name):
        return MANAGED, name
    if not sep or not name:
        raise ValueError(
            f"Unknown model '{model}'. Registered models: {MANAGER.names()}. "
            f"Or use '<provider>/<model>' with a provider from {AVAILABLE_PROVIDERS}."
        )
    if provider not in UPSTREAMS:
        raise ValueError(f"Unknown provider: '{provider}'. Available: {AVAILABLE_PROVIDERS}")
    return provider, name


async def close_upstreams() -> None:
    for upstream in UPSTREAMS.values():
        await upstream.aclose()
    await MANAGER.shutdown()


async def _managed_completion(name: str, body: dict) -> dict:
    resident = await MANAGER.acquire(name)
    try:
        body = {**MANAGER.request_defaults(resident.spec), **body}
        return await resident.backend.chat_completion(body)
    finally:
        await MANAGER.release()


async def _managed_stream(name: str, body: dict) -> AsyncIterator[bytes]:
    resident = await MANAGER.acquire(name)
    try:
        body = {**MANAGER.request_defaults(resident.spec), **body}
        stream = await resident.backend.open_stream(body)
    except BaseException:
        await MANAGER.release()
        raise

    async def relay() -> AsyncIterator[bytes]:
        # The model counts as in use until the stream ends or the client
        # disconnects, so a swap cannot stop it mid-answer.
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await MANAGER.release()

    return relay()


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

    if provider == MANAGED:
        response = await _managed_completion(model, body)
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

    if provider == MANAGED:
        return await _managed_stream(model, body)

    return await UPSTREAMS[provider].open_stream({**body, "model": model})

