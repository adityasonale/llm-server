"""
Pre-load VRAM estimation for local models.

Reads safetensors headers (a few KB) rather than the weights themselves, so a
model can be sized before anything touches the GPU.
"""

import glob
import json
import os
import struct
from utils.logger import get_logger

logger = get_logger(__name__)

# Bytes per element for the safetensors dtype names.
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

# Bytes per parameter after quantization, by CustomLLM's `quantize` setting.
_QUANTIZED_BYTES = {"4-bit": 0.5, "8-bit": 1.0, None: 2.0}

# bitsandbytes only quantizes Linear layers. Embeddings, the LM head, norms and
# biases stay at 16-bit — on a 7B model with a 152k vocab that is ~2GB that no
# quantization setting will reclaim, which is exactly what a naive
# params * 0.5 estimate misses.
_NEVER_QUANTIZED = ("embed_tokens", "lm_head", "norm", "bias")

# Headroom for CUDA context, activations and the KV cache. Deliberately coarse:
# the point is to catch "this cannot possibly fit", not to predict to the MB.
_OVERHEAD_GB = 1.5


def _read_header(path: str) -> dict:
    """Read a safetensors file's JSON header without reading any tensor data."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(header_len))


def estimate_vram_gb(model_dir: str, quantize: str | None = "4-bit") -> float | None:
    """
    Estimate the VRAM a model needs once loaded, in GB.

    Returns None when the model cannot be sized (no local safetensors files —
    e.g. a Hub repo that has not been downloaded yet), so callers can decide
    whether to proceed rather than being handed a wrong number.
    """
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    if not shards:
        logger.warning("Cannot size model at %s: no .safetensors files found", model_dir)
        return None

    quantized_params = 0
    full_precision_bytes = 0

    for shard in shards:
        try:
            header = _read_header(shard)

        except Exception as e:
            logger.warning("Could not read safetensors header from %s: %s", shard, e)
            return None

        for name, meta in header.items():
            if name == "__metadata__":
                continue

            numel = 1
            for dim in meta["shape"]:
                numel *= dim

            if any(key in name for key in _NEVER_QUANTIZED):
                full_precision_bytes += numel * _DTYPE_BYTES.get(meta["dtype"], 2)
            else:
                quantized_params += numel

    bytes_per_param = _QUANTIZED_BYTES.get(quantize, 2.0)
    weights_gb = (quantized_params * bytes_per_param + full_precision_bytes) / 1024 ** 3
    total_gb = weights_gb + _OVERHEAD_GB

    logger.info(
        "VRAM estimate for %s — weights: %.2fGB (quantize=%s), +%.1fGB overhead, total: %.2fGB",
        os.path.basename(model_dir.rstrip("/\\")), weights_gb, quantize, _OVERHEAD_GB, total_gb,
    )

    return total_gb


# ---------------------------------------------------------------------------
# Run with:  python -m providers.local.capacity      (from the repo root)
#
# Must be run as a module, not as a script: this file imports utils.logger
# absolutely, and `python providers/local/capacity.py` puts providers/local on
# sys.path instead of the repo root.
# ---------------------------------------------------------------------------

def _write_fake_shard(path: str, tensors: dict) -> None:
    """Write a safetensors file containing only a header — no tensor data."""
    header = {
        name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def _self_check() -> None:
    """Verify the estimate against hand-computed numbers, with no GPU or real weights."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # 1M params in a Linear layer (quantizable) + 100k in embeddings (not).
        _write_fake_shard(os.path.join(tmp, "model.safetensors"), {
            "model.layers.0.mlp.gate_proj.weight": ("BF16", [1024, 1024]),
            "model.embed_tokens.weight": ("BF16", [1000, 100]),
        })

        quantizable, kept = 1024 * 1024, 1000 * 100

        for quantize, per_param in (("4-bit", 0.5), ("8-bit", 1.0), (None, 2.0)):
            expected = (quantizable * per_param + kept * 2) / 1024 ** 3 + _OVERHEAD_GB
            actual = estimate_vram_gb(tmp, quantize=quantize)
            assert actual is not None and abs(actual - expected) < 1e-9, \
                f"quantize={quantize}: expected {expected}, got {actual}"
            print(f"  ok  quantize={str(quantize):6} -> {actual:.4f}GB")

        # Embeddings must NOT shrink with quantization: 4-bit and 8-bit differ
        # only by the Linear layer, which is the whole point of this estimator.
        four, eight = estimate_vram_gb(tmp, "4-bit"), estimate_vram_gb(tmp, "8-bit")
        assert abs((eight - four) - (quantizable * 0.5) / 1024 ** 3) < 1e-9
        print("  ok  quantization affects Linear layers only")

    # A directory with no safetensors files is unsizeable, not an error.
    with tempfile.TemporaryDirectory() as empty:
        assert estimate_vram_gb(empty) is None
        print("  ok  empty directory -> None")


if __name__ == "__main__":
    from settings import LOCAL_MODELS
    from utils.system import SystemMonitor

    print("\nself-check")
    _self_check()

    total, free = SystemMonitor().get_vram()
    print(f"\nGPU: {total:.2f}GB total, {free:.2f}GB free\n")

    name = "distilled-qwen-7b"
    path = LOCAL_MODELS[name].get("path")

    if not path or not os.path.isdir(path):
        print(f"  {name:24} path not present locally — cannot size")
    else:
        estimate = estimate_vram_gb(path, quantize="4-bit")

        if estimate is None:
            print(f"  {name:24} unsizeable")
        else:
            verdict = "fits" if estimate <= free else "WILL NOT FIT"
            print(f"  {name:24} {estimate:6.2f}GB   {verdict}")
    print()
