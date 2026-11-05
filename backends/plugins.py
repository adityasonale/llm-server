"""
Which backend plugins exist, and which one serves a given model on this machine.

Selection: a `backend:` pinned in models.yaml wins (and fails loudly if it
cannot run here). Otherwise, among plugins that handle the model's format and
can run on this hardware, the highest `priority` wins:

    gguf -> llamacpp (100)
    mlx  -> mlx (90)                           Apple Silicon only
    hf   -> mlx (90) > vllm (80) > transformers (10)
"""
import importlib

from backends.base import Backend, ModelSpec
from backends.hardware import Hardware
from backends.hf_transformers import TransformersBackend
from backends.llamacpp import LlamaCppBackend
from backends.mlx import MlxBackend
from backends.vllm import VllmBackend

BUILTIN: list[type[Backend]] = [LlamaCppBackend, MlxBackend, VllmBackend, TransformersBackend]


def load_plugins(extra: list[str] | None = None) -> dict[str, type[Backend]]:
    """Built-in plugins plus any listed as "module.path:ClassName" in models.yaml."""
    plugins = {cls.name: cls for cls in BUILTIN}
    for ref in extra or []:
        module_name, _, class_name = ref.partition(":")
        if not class_name:
            raise ValueError(f"Plugin '{ref}' must be written as 'module.path:ClassName'.")
        cls = getattr(importlib.import_module(module_name), class_name)
        if not (isinstance(cls, type) and issubclass(cls, Backend)) or not cls.name:
            raise ValueError(f"Plugin '{ref}' is not a Backend subclass with a name.")
        plugins[cls.name] = cls  # same name as a built-in replaces it
    return plugins


def select_backend(
    spec: ModelSpec, plugins: dict[str, type[Backend]], hw: Hardware, settings: dict[str, dict]
) -> type[Backend]:
    if spec.backend:
        cls = plugins.get(spec.backend)
        if cls is None:
            raise ValueError(f"Model '{spec.name}' pins unknown backend '{spec.backend}'. Known: {sorted(plugins)}")
        if spec.format not in cls.formats:
            raise ValueError(
                f"Backend '{cls.name}' cannot load {spec.format} models (it takes {sorted(cls.formats)})."
            )
        reason = cls.unavailable_reason(hw, settings.get(cls.name, {}))
        if reason:
            raise ValueError(f"Model '{spec.name}' pins backend '{cls.name}', which cannot run here: {reason}.")
        return cls

    candidates = sorted(
        (cls for cls in plugins.values() if spec.format in cls.formats),
        key=lambda cls: cls.priority, reverse=True,
    )
    reasons = []
    for cls in candidates:
        reason = cls.unavailable_reason(hw, settings.get(cls.name, {}))
        if reason is None:
            return cls
        reasons.append(f"{cls.name}: {reason}")

    if not candidates:
        raise ValueError(f"No backend plugin handles {spec.format} models.")
    raise ValueError(
        f"No backend can serve '{spec.name}' ({spec.format}) on this machine ({hw.describe()}). "
        + "; ".join(reasons)
    )
