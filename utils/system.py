import torch
import psutil
from utils.logger import get_logger

logger = get_logger(__name__)


class SystemMonitor:

    def get_ram(self) -> tuple[float, float]:
        mem = psutil.virtual_memory()
        free = mem.available / 1024 ** 3
        total = mem.total / 1024 ** 3
        return total, free

    def get_vram(self) -> tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        return total, free

    def get_torch_vram(self) -> tuple[float, float]:
        """
        VRAM held by torch's caching allocator: (reserved, allocated) in GB.

        `reserved` is what torch has taken from the driver; `allocated` is what
        is actually in use by live tensors. The gap between them is memory the
        driver reports as occupied but that torch can hand out immediately.
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        return reserved, allocated

    def get_available_vram(self) -> float:
        """
        VRAM this process can actually allocate, in GB.

        Driver-free memory alone understates this: after a model is unloaded,
        torch keeps the freed blocks reserved, so `mem_get_info` still counts
        them as used even though the next load can reuse them. Measured after
        unloading a 7B model, the driver reported 4.04GB free while ~2.9GB more
        sat in torch's cache, ready to use.
        """
        if not torch.cuda.is_available():
            return 0.0
        _, free = self.get_vram()
        reserved, allocated = self.get_torch_vram()
        return free + (reserved - allocated)

    def show_torch_vram(self) -> str:
        if not torch.cuda.is_available():
            return "TORCH: N/A (no local GPU)"
        reserved, allocated = self.get_torch_vram()
        return (f"TORCH: reserved {reserved:.2f}GB, allocated {allocated:.2f}GB, "
                f"cached {reserved - allocated:.2f}GB\t available: {self.get_available_vram():.2f}GB")

    def show_ram(self) -> str:
        total, free = self.get_ram()
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return f'RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

    def show_vram(self) -> str:
        if not torch.cuda.is_available():
            return 'VRAM: N/A (no local GPU)'
        total, free = self.get_vram()
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

    def log_memory(self, tag: str = ""):
        logger.info(
            "%s\n\t\t\t RAM:  %s\n\t\t\t VRAM: %s\n\t\t\t %s",
            tag,
            self.show_ram(),
            self.show_vram(),
            self.show_torch_vram()
        )