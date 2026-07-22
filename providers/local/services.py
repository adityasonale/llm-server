import asyncio
import gc
from pathlib import Path
from providers.base import LLMProvider
from providers.local.capacity import estimate_vram_gb
from providers.local.local_llm_server import CustomLLM
from providers.local.templates import resolve_prompt
from settings import LOCAL_MODELS
from utils.logger import get_logger
from utils.system import SystemMonitor

logger = get_logger(__name__)
monitor = SystemMonitor()


def _resolve_source(model: str, config: dict) -> tuple[str, bool]:
    """
    Decide where a model's weights come from.

    Returns (source, local_files_only) where source is either a filesystem path
    or a Hugging Face repo id. A local path is preferred when it exists, so a
    machine that already has the weights never hits the network.
    """
    path = config.get("path")
    repo_id = config.get("repo_id")

    if path and Path(path).is_dir():
        logger.info("Using local weights for %s: %s", model, path)
        return path, True

    if repo_id is None:
        raise ValueError(
            f"Model '{model}' has no usable source: path '{path}' does not exist and no repo_id is configured."
        )

    if path:
        logger.warning("Local path not found for %s (%s) — falling back to Hub: %s", model, path, repo_id)
    else:
        logger.info("Loading %s from Hub: %s", model, repo_id)

    return repo_id, False


class LocalService(LLMProvider):

    def __init__(self):
        self._model: CustomLLM | None = None
        self._current_model: str | None = None
        self._current_template: str | None = None
        self._lock = asyncio.Lock()
        # VRAM available to this process just before the resident model was
        # loaded — i.e. what will be available again once it is unloaded.
        self._available_vram_before_load: float | None = None

    def _check_capacity(self, model: str, source: str, local_files_only: bool) -> None:
        """
        Refuse a load that cannot fit, before the resident model is disturbed.

        Skips silently when the model cannot be sized (not yet downloaded) or
        when there is no CUDA device, rather than blocking on missing data.
        """
        if not local_files_only:
            logger.info("Skipping VRAM preflight for %s: weights are not on disk yet", model)
            return

        required = estimate_vram_gb(source)

        if required is None:
            logger.warning("Skipping VRAM preflight for %s: could not size the weights", model)
            return

        total, _ = monitor.get_vram()

        if total == 0.0:
            logger.info("Skipping VRAM preflight for %s: no CUDA device", model)
            return

        # A resident model's memory returns when it is unloaded, so compare
        # against the reading taken before it was loaded, not what is free now.
        if self._model is None or self._available_vram_before_load is None:
            available = monitor.get_available_vram()
        else:
            available = self._available_vram_before_load

        if required > available:
            raise ValueError(
                f"Model '{model}' needs ~{required:.2f}GB of VRAM but only {available:.2f}GB "
                f"will be available on a {total:.2f}GB device. Refusing to load; "
                f"the current model stays in memory."
            )

        logger.info(
            "VRAM preflight passed for %s — needs ~%.2fGB, %.2fGB available", model, required, available
        )

    def _load_model(self, model: str):
        config = LOCAL_MODELS.get(model)
        if config is None:
            raise ValueError(f"Unknown local model: '{model}'. Available: {list(LOCAL_MODELS.keys())}")

        if self._current_model == model:
            logger.info("Model already loaded: %s", model)
            return

        # Resolve and size the incoming model BEFORE unloading the current one:
        # a request that cannot fit must not cost us a working model.
        source, local_files_only = _resolve_source(model, config)
        self._check_capacity(model, source, local_files_only)

        if self._model is not None:
            previous = self._current_model
            logger.info("Unloading current model: %s", previous)
            monitor.log_memory("before unload")
            self._model.unload_model()
            self._model = None
            self._current_model = None
            self._current_template = None
            gc.collect()
            monitor.log_memory("after unload")

            # Unloading should hand back what was taken. Measured as available
            # to torch, not driver-free: torch keeps freed blocks reserved, so
            # the driver still counts them as used while they are reusable.
            available_after_unload = monitor.get_available_vram()
            if (self._available_vram_before_load is not None
                    and available_after_unload < self._available_vram_before_load - 0.25):
                logger.warning(
                    "VRAM not fully released after unloading %s — expected ~%.2fGB available, got %.2fGB",
                    previous, self._available_vram_before_load, available_after_unload,
                )

        self._available_vram_before_load = monitor.get_available_vram()

        logger.info("Loading local model: %s", model)
        monitor.log_memory("before load")
        self._model = CustomLLM(model_path=source, local_files_only=local_files_only)
        self._current_model = model
        self._current_template = config["prompt_template"]
        monitor.log_memory("after load")
        logger.info("Model loaded successfully: %s", model)

    async def fetch_response(self, system_prompt: str, user_prompt: str, model: str) -> str:
        async with self._lock:
            self._load_model(model)

            prompt = resolve_prompt(self._current_template, system_prompt, user_prompt)

            logger.info("Running local inference — model: %s", model)
            monitor.log_memory("before inference")
            response = self._model._call(prompt)
            monitor.log_memory("after inference")
            logger.info("Local inference completed — model: %s", model)

            return response