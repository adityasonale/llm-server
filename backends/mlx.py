"""
MLX and plain Hugging Face safetensors models on Apple Silicon, through
mlx-lm's OpenAI-compatible server. Needs `pip install mlx-lm` in the venv.
"""
import importlib.util
import sys

from backends.base import ModelSpec
from backends.hardware import Hardware
from backends.server import ServerBackend

_OVERHEAD_GB = 1.0


class MlxBackend(ServerBackend):
    name = "mlx"
    formats = frozenset({"mlx", "hf"})
    priority = 90
    # mlx_lm.server has no /health on every version; /v1/models is on all of them.
    health_path = "/v1/models"

    @classmethod
    def unavailable_reason(cls, hw: Hardware, settings: dict) -> str | None:
        if not hw.apple_silicon:
            return "MLX needs Apple Silicon"
        if importlib.util.find_spec("mlx_lm") is None:
            return "mlx-lm is not installed (pip install mlx-lm)"
        return None

    def command(self, spec: ModelSpec, host: str, port: int) -> list[str]:
        python = self.settings.get("python") or sys.executable
        return [python, "-m", "mlx_lm.server", "--model", spec.path, "--host", host, "--port", str(port)]

    def upstream_model(self, spec: ModelSpec) -> str:
        # mlx_lm.server loads whatever `model` names, per request. This value
        # means "the model given on the command line", so nothing is reloaded.
        return "default_model"

    def estimate_gb(self, spec: ModelSpec, hw: Hardware) -> float | None:
        return spec.size_bytes / 1024 ** 3 + _OVERHEAD_GB

    def budget_gb(self, hw: Hardware) -> float | None:
        return hw.ram_gb * 0.75
