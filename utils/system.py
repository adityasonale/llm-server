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
            "%s\n\t\t\t RAM:  %s\n\t\t\t VRAM: %s",
            tag,
            self.show_ram(),
            self.show_vram()
        )