"""GGUF models through llama.cpp's llama-server (Metal, CUDA or CPU — whatever the binary was built for)."""
import shutil

from backends.base import ModelSpec
from backends.hardware import Hardware
from backends.server import ServerBackend

# KV cache, compute buffers and context. Coarse on purpose, like capacity.py.
_OVERHEAD_GB = 1.0


def _binary(settings: dict) -> str | None:
    return settings.get("binary") or shutil.which("llama-server")


class LlamaCppBackend(ServerBackend):
    name = "llamacpp"
    formats = frozenset({"gguf"})
    priority = 100

    @classmethod
    def unavailable_reason(cls, hw: Hardware, settings: dict) -> str | None:
        if _binary(settings) is None:
            return "llama-server not found: set backends.llamacpp.binary in models.yaml or put it on PATH"
        return None

    def command(self, spec: ModelSpec, host: str, port: int) -> list[str]:
        # Nothing hardware-specific here: recent llama-server defaults to --fit
        # on (GPU layers and context sized to the device) and --jinja (needed
        # for tool calls). Anything else comes from the model's `args`.
        return [_binary(self.settings), "-m", spec.path, "--host", host, "--port", str(port)]

    def estimate_gb(self, spec: ModelSpec, hw: Hardware) -> float | None:
        return spec.size_bytes / 1024 ** 3 + _OVERHEAD_GB

    def budget_gb(self, hw: Hardware) -> float | None:
        if hw.apple_silicon:
            # macOS lets the GPU wire ~75% of unified memory by default.
            return hw.ram_gb * 0.75
        # Layers that do not fit in VRAM are offloaded to system RAM.
        return hw.vram_gb + hw.ram_gb * 0.5
