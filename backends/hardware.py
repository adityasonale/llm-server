"""
What this machine can run, detected without importing torch.

Plugins use it to decide whether they can serve here at all (MLX needs Apple
Silicon, vLLM needs CUDA) and how much memory a model may use.
"""
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Hardware:
    os: str                 # "darwin", "linux", "windows"
    apple_silicon: bool     # unified memory: GPU and CPU share `ram_gb`
    cuda: bool
    vram_gb: float          # total across NVIDIA GPUs; 0 without CUDA
    ram_gb: float

    def describe(self) -> str:
        if self.apple_silicon:
            return f"Apple Silicon, {self.ram_gb:.0f}GB unified memory"
        if self.cuda:
            return f"{self.os}, NVIDIA {self.vram_gb:.1f}GB VRAM, {self.ram_gb:.0f}GB RAM"
        return f"{self.os}, CPU only, {self.ram_gb:.0f}GB RAM"


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except ImportError:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3


def _nvidia_vram_gb() -> float:
    """Total VRAM via nvidia-smi; 0 when there is no NVIDIA driver."""
    if shutil.which("nvidia-smi") is None:
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return sum(float(line) for line in out.split() if line.strip()) / 1024
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        logger.warning("nvidia-smi present but unreadable: %s", e)
        return 0.0


@lru_cache(maxsize=1)
def detect() -> Hardware:
    system = platform.system().lower()
    vram = _nvidia_vram_gb()
    hw = Hardware(
        os=system,
        apple_silicon=system == "darwin" and platform.machine() == "arm64",
        cuda=vram > 0,
        vram_gb=vram,
        ram_gb=_ram_gb(),
    )
    logger.info("Hardware: %s", hw.describe())
    return hw
