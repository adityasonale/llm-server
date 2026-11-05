"""
Serve any model by name from a path in models.yaml.

For each request naming a registered model, the manager:
  1. detects the model's format from its path   (backends/formats.py)
  2. picks a backend plugin for format + hardware (backends/plugins.py)
  3. checks it can fit, BEFORE touching the resident model
  4. stops the resident backend and starts the new one, if it is a different model
  5. hands the running backend to the caller, counted as in use

One model is resident at a time, like LocalService: one GPU (or one pool of
unified memory) cannot hold two of these models. Requests for the resident
model run concurrently; the backend (llama-server etc.) queues them itself.
A swap waits for in-flight requests on the old model to finish first.

models.yaml is re-read when it changes on disk, so adding a model needs no
restart. Changing a resident model's entry restarts it on its next request.
"""
import asyncio
import json
import os
from dataclasses import dataclass

import yaml

from backends.base import Backend, ModelSpec
from backends.formats import detect_format
from backends.hardware import Hardware, detect
from backends.plugins import load_plugins, select_backend
from providers.base import ProviderOverloadedError
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_FILE = os.getenv("MODELS_FILE", "models.yaml")
# How long a swap waits for requests on the old model to finish.
DRAIN_TIMEOUT_SECONDS = 120


def _freeze(value):
    """Make yaml values hashable, so ModelSpec equality can tell when an entry changed."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


_KNOWN_KEYS = {"path", "backend", "args", "defaults", "port", "startup_timeout"}


@dataclass
class Resident:
    spec: ModelSpec
    backend: Backend


class ModelManager:

    def __init__(self, models_file: str = MODELS_FILE, hardware: Hardware | None = None):
        self._file = models_file
        self._hw = hardware
        self._mtime: float | None = None
        self._entries: dict[str, dict] = {}
        self._backend_settings: dict[str, dict] = {}
        self._plugins: dict = load_plugins()

        self._cond = asyncio.Condition()
        self._resident: Resident | None = None
        self._in_use = 0
        self._swapping = False

    @property
    def hardware(self) -> Hardware:
        if self._hw is None:
            self._hw = detect()
        return self._hw

    # -- registry ----------------------------------------------------------

    def _reload(self) -> None:
        try:
            mtime = os.path.getmtime(self._file)
        except OSError:
            if self._entries:
                logger.warning("%s disappeared; no registered models", self._file)
            self._entries, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return

        with open(self._file, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        models = raw.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError(f"{self._file}: 'models' must be a mapping of name -> entry.")
        for name, entry in models.items():
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(f"{self._file}: model '{name}' needs at least a 'path'.")

        self._plugins = load_plugins(raw.get("plugins"))
        self._backend_settings = raw.get("backends") or {}
        self._entries = {str(name): entry for name, entry in models.items()}
        self._mtime = mtime
        logger.info("Loaded %d model(s) from %s: %s", len(self._entries), self._file, list(self._entries))

    def names(self) -> list[str]:
        self._reload()
        return list(self._entries)

    def has(self, name: str) -> bool:
        self._reload()
        return name in self._entries

    def spec(self, name: str) -> ModelSpec:
        self._reload()
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"Unknown model '{name}'. Registered in {self._file}: {list(self._entries)}")
        detected = detect_format(str(entry["path"]))
        return ModelSpec(
            name=name,
            path=detected.path,
            format=detected.format,
            size_bytes=detected.size_bytes,
            backend=entry.get("backend"),
            args=tuple(str(a) for a in entry.get("args") or ()),
            defaults=json.dumps(entry.get("defaults") or {}, sort_keys=True),
            options=_freeze({k: v for k, v in entry.items() if k not in _KNOWN_KEYS}),
            port=entry.get("port"),
            startup_timeout=float(entry.get("startup_timeout", 300)),
        )

    def describe(self) -> list[dict]:
        """For GET /v1/models: each registered model, its format, and the backend it would use here."""
        rows = []
        for name in self.names():
            row = {"id": name, "object": "model", "owned_by": "llm-server"}
            try:
                spec = self.spec(name)
                row["format"] = spec.format
                row["backend"] = select_backend(spec, self._plugins, self.hardware, self._backend_settings).name
            except ValueError as e:
                row["error"] = str(e)
            row["loaded"] = self._resident is not None and self._resident.spec.name == name
            rows.append(row)
        return rows

    @staticmethod
    def request_defaults(spec: ModelSpec) -> dict:
        return json.loads(spec.defaults)

    # -- lifecycle ---------------------------------------------------------

    async def acquire(self, name: str) -> Resident:
        """
        Return the running model for `name`, starting or swapping to it if needed.
        Every successful acquire must be paired with release().
        """
        spec = self.spec(name)

        async with self._cond:
            while self._swapping:
                await self._cond.wait()
            if self._resident is not None and self._resident.spec == spec:
                self._in_use += 1
                return self._resident

            # Claim the swap. New requests (for any model) now wait, so the
            # resident model's in-flight requests can drain.
            self._swapping = True
            if self._in_use:
                logger.info("Waiting for %d in-flight request(s) before swapping to %s", self._in_use, name)
                try:
                    await asyncio.wait_for(self._cond.wait_for(lambda: self._in_use == 0), DRAIN_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    self._swapping = False
                    self._cond.notify_all()
                    raise ProviderOverloadedError(
                        f"Timed out after {DRAIN_TIMEOUT_SECONDS}s waiting for requests on "
                        f"'{self._resident.spec.name}' to finish before loading '{name}'."
                    )

        # The (slow) load happens outside the lock; `_swapping` holds everyone else off.
        try:
            resident = await self._swap_to(spec)
        except BaseException:
            async with self._cond:
                self._swapping = False
                self._cond.notify_all()
            raise

        async with self._cond:
            self._swapping = False
            self._in_use += 1
            self._cond.notify_all()
        return resident

    async def release(self) -> None:
        async with self._cond:
            self._in_use -= 1
            self._cond.notify_all()

    async def _swap_to(self, spec: ModelSpec) -> Resident:
        hw = self.hardware
        cls = select_backend(spec, self._plugins, hw, self._backend_settings)
        backend = cls(self._backend_settings.get(cls.name, {}))

        # Size the incoming model BEFORE stopping the resident one: a model
        # that cannot fit must not cost us a working one. Compared against the
        # machine's total budget, not what is free now, because the resident
        # model's memory comes back when it stops.
        needed, budget = backend.estimate_gb(spec, hw), backend.budget_gb(hw)
        if needed is not None and budget is not None and needed > budget:
            raise ValueError(
                f"Model '{spec.name}' needs ~{needed:.1f}GB but {cls.name} can use at most ~{budget:.1f}GB "
                f"on this machine ({hw.describe()}). Refusing to load; the current model stays loaded."
            )

        if self._resident is not None:
            old = self._resident
            logger.info("Swapping %s (%s) -> %s (%s)", old.spec.name, type(old.backend).name, spec.name, cls.name)
            self._resident = None
            await old.backend.stop()

        logger.info(
            "Loading %s: %s via %s (%s, %.1fGB on disk)",
            spec.name, spec.path, cls.name, spec.format, spec.size_bytes / 1024 ** 3,
        )
        await backend.start(spec)  # a failed start cleans up after itself
        self._resident = Resident(spec, backend)
        return self._resident

    async def shutdown(self) -> None:
        if self._resident is not None:
            await self._resident.backend.stop()
            self._resident = None


MANAGER = ModelManager()


# ---------------------------------------------------------------------------
# Self-check: python -m model_manager   (no real models or GPU needed)
# ---------------------------------------------------------------------------

def _self_check() -> None:
    import tempfile
    from collections.abc import AsyncIterator

    from backends import formats

    formats._self_check()

    events: list[str] = []

    class FakeBackend(Backend):
        name = "fake"
        formats = frozenset({"gguf"})
        priority = 1000

        async def start(self, spec):
            events.append(f"start {spec.name}")
            if spec.option("fail"):
                raise RuntimeError("boom")
            self.spec = spec

        async def stop(self):
            events.append(f"stop {self.spec.name}")

        async def chat_completion(self, body):
            await asyncio.sleep(0.05)
            return {"model": self.spec.name}

        async def open_stream(self, body) -> AsyncIterator[bytes]:
            raise NotImplementedError

        def estimate_gb(self, spec, hw):
            return spec.size_bytes / 1024 ** 3

        def budget_gb(self, hw):
            return 1.0  # 1GB machine

    hw = Hardware(os="test", apple_silicon=False, cuda=False, vram_gb=0, ram_gb=1)

    with tempfile.TemporaryDirectory() as tmp:
        for n, size in (("a", 10), ("b", 10), ("huge", 0)):
            with open(os.path.join(tmp, f"{n}.gguf"), "wb") as fh:
                fh.write(b"x" * size)
        # A 2GB "model" without writing 2GB: a sparse file.
        with open(os.path.join(tmp, "huge.gguf"), "wb") as fh:
            fh.truncate(2 * 1024 ** 3)

        models_file = os.path.join(tmp, "models.yaml")
        with open(models_file, "w") as fh:
            yaml.safe_dump({"models": {
                "a": {"path": os.path.join(tmp, "a.gguf"), "defaults": {"temperature": 0.1, "stop": ["x"]}},
                "b": {"path": os.path.join(tmp, "b.gguf")},
                "huge": {"path": os.path.join(tmp, "huge.gguf")},
                "broken": {"path": os.path.join(tmp, "a.gguf"), "fail": True},
            }}, fh)

        manager = ModelManager(models_file, hardware=hw)
        manager._reload()
        manager._plugins = {"fake": FakeBackend}

        assert manager.request_defaults(manager.spec("a")) == {"temperature": 0.1, "stop": ["x"]}
        assert manager.spec("broken").option("fail") is True

        async def use(name):
            r = await manager.acquire(name)
            try:
                return await r.backend.chat_completion({})
            finally:
                await manager.release()

        async def scenario():
            # Concurrent requests for one model share a single start.
            assert [r["model"] for r in await asyncio.gather(use("a"), use("a"), use("a"))] == ["a"] * 3
            assert events == ["start a"], events

            # A swap waits for in-flight work, then stops before starting.
            await asyncio.gather(use("a"), use("b"))
            assert events == ["start a", "stop a", "start b"], events
            assert manager._in_use == 0

            # Too big: refused with the resident model untouched.
            try:
                await use("huge")
                raise AssertionError("expected ValueError")
            except ValueError as e:
                assert "Refusing" in str(e)
            assert events[-1] == "start b" and manager._resident.spec.name == "b"

            # A failed start leaves nothing resident and does not wedge the manager.
            try:
                await use("broken")
                raise AssertionError("expected RuntimeError")
            except RuntimeError:
                pass
            assert manager._resident is None and not manager._swapping
            assert (await use("a"))["model"] == "a"

            # Unknown names are a ValueError (-> 400).
            try:
                await manager.acquire("nope")
                raise AssertionError("expected ValueError")
            except ValueError:
                pass

            await manager.shutdown()

        asyncio.run(scenario())
    print("  ok  manager: shared start, drain-then-swap, too-big refusal, failed start, unknown name")

    # Selection by format and hardware with the real plugins.
    from backends.plugins import BUILTIN
    plugins = {cls.name: cls for cls in BUILTIN}
    mac = Hardware(os="darwin", apple_silicon=True, cuda=False, vram_gb=0, ram_gb=16)
    spec = ModelSpec(name="x", path="/x", format="mlx", size_bytes=0)
    try:
        select_backend(spec, plugins, hw, {})
        raise AssertionError("mlx must not be servable off Apple Silicon")
    except ValueError as e:
        assert "Apple Silicon" in str(e), e
    gguf = ModelSpec(name="x", path="/x", format="gguf", size_bytes=0)
    assert select_backend(gguf, plugins, mac, {"llamacpp": {"binary": "/bin/sh"}}) is plugins["llamacpp"]
    print("  ok  selection: mlx refused off Apple Silicon, gguf -> llamacpp")


if __name__ == "__main__":
    _self_check()
