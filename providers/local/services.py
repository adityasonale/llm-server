import asyncio
import gc
from pathlib import Path
from providers.base import LLMProvider, ProviderOverloadedError
from providers.local.capacity import estimate_vram_gb
from providers.local.local_llm_server import CustomLLM
from providers.local.templates import resolve_prompt
from settings import LOCAL_MODELS
from utils.logger import get_logger
from utils.system import SystemMonitor

logger = get_logger(__name__)
monitor = SystemMonitor()

# Only one request can use the GPU at a time, so everything else waits. These
# bound that wait instead of letting it grow without limit.
#
# The counter covers the running request too, not just those queued behind it:
# 3 means one running plus two waiting. A 7B inference measured ~33s, so the
# last accepted request waits roughly 66s.
MAX_INFLIGHT_REQUESTS = 3
QUEUE_TIMEOUT_SECONDS = 120


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
        # Requests waiting for the lock. Not the same as lock.locked(): this is how many are queued behind whoever is currently running.
        self._waiting = 0
        # VRAM available to this process just before the resident model was loaded — i.e. what will be available again once it is unloaded.
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
            # Deliberately NOT ProviderOverloadedError: this does not clear on
            # its own, so telling the client to retry would send it into a
            # futile loop. The model simply cannot be served by this device —
            # a 400, like asking for a model that does not exist.
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

    async def _acquire(self) -> None:
        """
        Take the GPU lock, refusing rather than queueing without limit.

        Rejects immediately when the queue is already at depth, and gives up
        after QUEUE_TIMEOUT_SECONDS otherwise.
        """
        if self._waiting >= MAX_INFLIGHT_REQUESTS:
            logger.warning("Refusing request — %d already in flight", self._waiting)
            raise ProviderOverloadedError(
                f"Local provider is busy: {self._waiting} requests already in flight."
            )

        self._waiting += 1
        try:
            if self._waiting > 1 or self._lock.locked():
                logger.info("Request queued — position %d", self._waiting)

            # shield() so a timeout cannot cancel an acquisition that has
            # already been granted, which would leave the lock held forever.
            acquisition = asyncio.ensure_future(self._lock.acquire())
            try:
                await asyncio.wait_for(asyncio.shield(acquisition), timeout=QUEUE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                # The acquire is still pending; release it if it later succeeds.
                acquisition.add_done_callback(
                    lambda task: self._lock.release() if not task.cancelled() and not task.exception() else None
                )
                logger.warning("Request timed out after %ss waiting for the GPU", QUEUE_TIMEOUT_SECONDS)
                raise ProviderOverloadedError(
                    f"Timed out after {QUEUE_TIMEOUT_SECONDS}s waiting for the local provider."
                )
        finally:
            self._waiting -= 1

    async def fetch_response(self, system_prompt: str, user_prompt: str, model: str) -> str:
        await self._acquire()
        try:
            # Both the load and the inference are synchronous and take tens of
            # seconds. Run them on a worker thread: on the event loop they would
            # freeze the whole server, including /health and the Groq provider,
            # and would stop any asyncio timeout from ever firing.
            await asyncio.to_thread(self._load_model, model)

            prompt = resolve_prompt(self._current_template, system_prompt, user_prompt)

            logger.info("Running local inference — model: %s", model)
            monitor.log_memory("before inference")
            response = await asyncio.to_thread(self._model._call, prompt)
            monitor.log_memory("after inference")
            logger.info("Local inference completed — model: %s", model)

            return response
        finally:
            self._lock.release()