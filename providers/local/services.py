import asyncio
import gc
from pathlib import Path
from providers.base import LLMProvider
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

    def _load_model(self, model: str):
        config = LOCAL_MODELS.get(model)
        if config is None:
            raise ValueError(f"Unknown local model: '{model}'. Available: {list(LOCAL_MODELS.keys())}")

        if self._current_model == model:
            logger.info("Model already loaded: %s", model)
            return

        if self._model is not None:
            logger.info("Unloading current model: %s", self._current_model)
            monitor.log_memory("before unload")
            self._model.unload_model()
            self._model = None
            self._current_model = None
            self._current_template = None
            gc.collect()
            monitor.log_memory("after unload")

        source, local_files_only = _resolve_source(model, config)

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