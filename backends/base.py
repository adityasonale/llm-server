"""
The plugin contract for model loading.

A backend plugin knows how to serve some model formats on some hardware.
`model_manager.py` picks one per model (format + hardware + priority, or a
`backend:` pinned in models.yaml), starts it on demand, and sends requests to
it. To add a backend, subclass `Backend` and either add it to
`backends/plugins.py::BUILTIN` or list it under `plugins:` in models.yaml as
"module.path:ClassName".

Error conventions, matching the rest of the server:
    ValueError      -> 400  the request can never be served (bad path, wrong hardware, too big)
    UpstreamError   -> 502  the backend failed to start or crashed
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from backends.hardware import Hardware


@dataclass(frozen=True)
class ModelSpec:
    """One entry of models.yaml, with its format already detected."""
    name: str
    path: str                   # resolved: the .gguf file or the model directory
    format: str                 # see backends/formats.py
    size_bytes: int
    backend: str | None = None  # pinned plugin name; None = choose automatically
    args: tuple = ()            # extra CLI args for server backends
    defaults: str = "{}"        # sampling defaults as a JSON object; request values override them
    options: tuple = ()         # any other keys, as sorted (key, value) pairs, for the plugin
    port: int | None = None     # fixed port for server backends; None = pick a free one
    startup_timeout: float = 300.0

    def option(self, key: str, default=None):
        return dict(self.options).get(key, default)


class Backend(ABC):
    """
    One running model. A fresh instance is created per load, so instance
    state (process, client) belongs to exactly one model.

    Class attributes describe the plugin and are read before any instance exists.
    """
    name: str = ""
    formats: frozenset[str] = frozenset()
    # Higher wins when several plugins can serve a format on this machine.
    priority: int = 0
    supports_tools: bool = True

    def __init__(self, settings: dict | None = None):
        # Machine-specific plugin settings from models.yaml `backends.<name>`.
        self.settings = settings or {}

    @classmethod
    def unavailable_reason(cls, hw: Hardware, settings: dict) -> str | None:
        """None if this plugin can run here; otherwise a reason to show the user."""
        return None

    def estimate_gb(self, spec: ModelSpec, hw: Hardware) -> float | None:
        """Memory the model needs once loaded; None = cannot tell, skip the check."""
        return None

    def budget_gb(self, hw: Hardware) -> float | None:
        """
        Most memory a model may use on this machine with nothing else loaded;
        None = no check. Deliberately coarse: it catches "cannot possibly fit",
        it does not predict to the MB.
        """
        return None

    @abstractmethod
    async def start(self, spec: ModelSpec) -> None:
        """Load the model and return once it can take requests."""

    @abstractmethod
    async def stop(self) -> None:
        """Free everything start() took. Must be safe to call after a failed start."""

    @abstractmethod
    async def chat_completion(self, body: dict) -> dict:
        """Non-streaming OpenAI chat completion."""

    @abstractmethod
    async def open_stream(self, body: dict) -> AsyncIterator[bytes]:
        """
        Start a streaming completion; return an iterator of raw SSE bytes.
        Errors must be raised here, before the first byte.
        """
