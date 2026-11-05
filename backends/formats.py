"""
Work out what kind of model a path points at, from the files alone.

Formats:
    gguf  — a .gguf file, or a directory holding one (split files: the first shard)
    mlx   — a Hugging Face-style directory quantized by mlx-lm
            (config.json has a top-level "quantization": {"bits", "group_size"})
    hf    — a Hugging Face directory: config.json plus *.safetensors or *.bin

Nothing here loads weights; at most it reads config.json.
"""
import json
import os
import re
from dataclasses import dataclass

FORMATS = ("gguf", "mlx", "hf")

# llama.cpp split files: model-00001-of-00003.gguf. llama-server is pointed at
# the first shard and finds the rest itself.
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


@dataclass(frozen=True)
class DetectedModel:
    format: str
    path: str               # what to hand the backend: the .gguf file, or the directory
    size_bytes: int         # weights on disk, for the capacity check


def _gguf_in_dir(directory: str) -> list[str]:
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".gguf"))
    # mmproj files are vision projectors that accompany a model, not a model.
    return [f for f in files if not f.lower().startswith("mmproj")]


def _gguf_size(path: str) -> int:
    """Size of a GGUF model, counting every shard of a split model."""
    match = _SPLIT.search(path)
    if not match:
        return os.path.getsize(path)
    prefix = path[: match.start()]
    total = int(match.group(2))
    return sum(
        os.path.getsize(f"{prefix}-{i:05d}-of-{total:05d}.gguf")
        for i in range(1, total + 1)
        if os.path.exists(f"{prefix}-{i:05d}-of-{total:05d}.gguf")
    )


def detect_format(path: str) -> DetectedModel:
    """Raise ValueError (-> HTTP 400) when the path is missing or not a known format."""
    path = os.path.expanduser(os.path.expandvars(path))

    if os.path.isfile(path):
        if path.lower().endswith(".gguf"):
            return DetectedModel("gguf", path, _gguf_size(path))
        raise ValueError(
            f"Unsupported model file '{os.path.basename(path)}'. A single file must be a .gguf; "
            f"other formats are loaded from their directory. Supported formats: {', '.join(FORMATS)}."
        )

    if not os.path.isdir(path):
        raise ValueError(f"Model path does not exist: {path}")

    ggufs = _gguf_in_dir(path)
    if ggufs:
        firsts = [f for f in ggufs if not _SPLIT.search(f) or _SPLIT.search(f).group(1) == "00001"]
        if len(firsts) > 1:
            raise ValueError(
                f"{path} holds several GGUF models ({', '.join(firsts)}). Point the path at one file."
            )
        chosen = os.path.join(path, firsts[0])
        return DetectedModel("gguf", chosen, _gguf_size(chosen))

    config_path = os.path.join(path, "config.json")
    weights = [f for f in os.listdir(path) if f.endswith((".safetensors", ".bin"))]
    if os.path.isfile(config_path) and weights:
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        size = sum(os.path.getsize(os.path.join(path, f)) for f in weights)
        # mlx-lm writes its own top-level "quantization" block; a transformers
        # quantized model uses "quantization_config" instead.
        quant = config.get("quantization")
        if isinstance(quant, dict) and "bits" in quant:
            return DetectedModel("mlx", path, size)
        return DetectedModel("hf", path, size)

    raise ValueError(
        f"Could not recognise a model at {path}: expected a .gguf file, or a directory with "
        f"config.json and *.safetensors/*.bin weights. Supported formats: {', '.join(FORMATS)}."
    )


def _self_check() -> None:
    import tempfile

    def touch(p: str, data: bytes = b"x") -> None:
        with open(p, "wb") as fh:
            fh.write(data)

    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.gguf")
        touch(f, b"1234")
        assert detect_format(f) == DetectedModel("gguf", f, 4)

        d = os.path.join(tmp, "split")
        os.mkdir(d)
        for i in (1, 2):
            touch(os.path.join(d, f"m-0000{i}-of-00002.gguf"), b"12")
        touch(os.path.join(d, "mmproj-f16.gguf"))
        got = detect_format(d)
        assert got.format == "gguf" and got.path.endswith("m-00001-of-00002.gguf") and got.size_bytes == 4, got

        hf = os.path.join(tmp, "hf")
        os.mkdir(hf)
        touch(os.path.join(hf, "config.json"), b'{"model_type": "qwen3"}')
        touch(os.path.join(hf, "model.safetensors"), b"123")
        assert detect_format(hf) == DetectedModel("hf", hf, 3)

        mlx = os.path.join(tmp, "mlx")
        os.mkdir(mlx)
        touch(os.path.join(mlx, "config.json"), b'{"quantization": {"bits": 4, "group_size": 64}}')
        touch(os.path.join(mlx, "model.safetensors"))
        assert detect_format(mlx).format == "mlx"

        for bad in (os.path.join(tmp, "missing"), os.path.join(hf, "config.json")):
            try:
                detect_format(bad)
                raise AssertionError(f"expected ValueError for {bad}")
            except ValueError:
                pass
    print("  ok  formats: gguf file, split gguf dir, hf dir, mlx dir, bad paths")


if __name__ == "__main__":
    _self_check()
