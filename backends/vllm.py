"""
Hugging Face safetensors models on NVIDIA GPUs through vLLM's OpenAI server.
Linux (or WSL) only, and needs `pip install vllm`.

NOT YET VERIFIED: written without a CUDA machine to test on. Tool calling in
vLLM needs per-model flags, e.g. args: ["--enable-auto-tool-choice",
"--tool-call-parser", "hermes"] in models.yaml.
"""
import importlib.util
import sys

from backends.base import ModelSpec
from backends.hardware import Hardware
from backends.server import ServerBackend

_OVERHEAD_GB = 1.0


class VllmBackend(ServerBackend):
    name = "vllm"
    formats = frozenset({"hf"})
    priority = 80

    @classmethod
    def unavailable_reason(cls, hw: Hardware, settings: dict) -> str | None:
        if not hw.cuda:
            return "vLLM needs an NVIDIA GPU"
        if hw.os != "linux":
            return "vLLM runs on Linux only (use WSL on Windows)"
        if importlib.util.find_spec("vllm") is None:
            return "vllm is not installed (pip install vllm)"
        return None

    def command(self, spec: ModelSpec, host: str, port: int) -> list[str]:
        python = self.settings.get("python") or sys.executable
        return [
            python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.path, "--served-model-name", spec.name,
            "--host", host, "--port", str(port),
        ]

    def estimate_gb(self, spec: ModelSpec, hw: Hardware) -> float | None:
        return spec.size_bytes / 1024 ** 3 + _OVERHEAD_GB

    def budget_gb(self, hw: Hardware) -> float | None:
        # vLLM keeps the whole model on the GPU; no CPU offload by default.
        return hw.vram_gb
