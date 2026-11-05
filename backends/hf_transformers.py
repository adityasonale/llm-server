"""
Hugging Face safetensors models in-process through transformers — the
fallback when no server backend can take an `hf` model here.

It reuses the LocalService that /generate and `local/` use, so all three share
one GPU lock, one admission queue and one resident torch model. Same limits as
`local/`: no tool calls, and a stream arrives as a single chunk.
"""
import importlib.util
from collections.abc import AsyncIterator

from backends.base import Backend, ModelSpec
from backends.hardware import Hardware


def _local_service():
    from router import PROVIDERS  # deferred: router imports torch
    return PROVIDERS["local"]


class TransformersBackend(Backend):
    name = "transformers"
    formats = frozenset({"hf"})
    priority = 10
    supports_tools = False

    def __init__(self, settings: dict | None = None):
        super().__init__(settings)
        self._spec: ModelSpec | None = None

    @classmethod
    def unavailable_reason(cls, hw: Hardware, settings: dict) -> str | None:
        if importlib.util.find_spec("transformers") is None:
            return "transformers is not installed (pip install -r requirements-local.txt)"
        return None

    def estimate_gb(self, spec: ModelSpec, hw: Hardware) -> float | None:
        # LocalService runs its own safetensors-header preflight on CUDA,
        # which accounts for quantization; nothing to add here.
        return None

    async def start(self, spec: ModelSpec) -> None:
        # bitsandbytes 4-bit needs CUDA; elsewhere default to unquantized fp16.
        from backends.hardware import detect
        default_quantize = "4-bit" if detect().cuda else None
        _local_service().register(spec.name, {
            "path": spec.path,
            "prompt_template": spec.option("prompt_template"),
            "quantize": spec.option("quantize", default_quantize),
        })
        self._spec = spec
        # The weights load on the first request, inside LocalService's lock.

    async def stop(self) -> None:
        await _local_service().unload()

    async def _generate(self, body: dict) -> str:
        from chat import _local_request  # deferred: chat imports the model manager
        messages, kwargs = _local_request(body)
        return await _local_service().chat(self._spec.name, messages, **kwargs)

    async def chat_completion(self, body: dict) -> dict:
        from chat import _local_completion
        return _local_completion(body["model"], await self._generate(body))

    async def open_stream(self, body: dict) -> AsyncIterator[bytes]:
        from chat import _local_stream
        return _local_stream(body["model"], await self._generate(body))
