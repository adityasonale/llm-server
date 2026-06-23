import asyncio
import gc
from providers.base import LLMProvider
from providers.local.local_llm_server import CustomLLM
from providers.local.templates import resolve_prompt
from settings import LOCAL_MODELS
from utils.logger import get_logger
from utils.system import SystemMonitor

logger = get_logger(__name__)
monitor = SystemMonitor()


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

        logger.info("Loading local model: %s", model)
        monitor.log_memory("before load")
        self._model = CustomLLM(model_path=config["path"])
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