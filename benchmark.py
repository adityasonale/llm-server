"""
Repeatable measurement of the local provider.

Records cold-start time, inference latency and PEAK VRAM across a range of
prompt sizes, appending one JSON object per measurement to
benchmarks/results.jsonl so runs can be compared over time.

Peak memory comes from torch's own counters rather than sampling before and
after, so a transient spike during generation cannot be missed.

Run from the repo root:
    python benchmark.py                       # default model, default sizes
    python benchmark.py --model minichat-1.5-3b
    python benchmark.py --prompt-tokens 64 512 2000 --max-new-tokens 128
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from providers.local.capacity import estimate_vram_gb
from providers.local.services import LocalService, _resolve_source
from providers.local.templates import resolve_prompt
from settings import LOCAL_MODELS
from utils.logger import get_logger
from utils.system import SystemMonitor

logger = get_logger(__name__)
monitor = SystemMonitor()

RESULTS = Path("benchmarks/results.jsonl")


def _git_commit() -> str | None:
    """Record which revision produced a measurement; results are useless without it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _make_prompt(tokenizer, n_tokens: int) -> str:
    """Build a prompt that is close to exactly n_tokens long for this tokenizer."""
    filler = "The quick brown fox jumps over the lazy dog. "
    text = filler * max(1, n_tokens // 8)
    ids = tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids)


def _record(row: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilled-qwen-7b", choices=list(LOCAL_MODELS))
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 512, 1024, 2000])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("No CUDA device — these numbers would be meaningless. Aborting.")
        return

    config = LOCAL_MODELS[args.model]
    source, local_files_only = _resolve_source(args.model, config)
    estimated = estimate_vram_gb(source)

    run = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "estimated_vram_gb": round(estimated, 3) if estimated else None,
        "max_new_tokens": args.max_new_tokens,
    }

    print(f"\n{run['gpu']} | {args.model} | estimate {run['estimated_vram_gb']}GB\n")

    # --- cold start -------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    service = LocalService()

    started = time.perf_counter()
    service._load_model(args.model)
    load_seconds = time.perf_counter() - started

    weights_allocated = torch.cuda.memory_allocated() / 1024 ** 3
    weights_reserved = torch.cuda.memory_reserved() / 1024 ** 3

    _record({**run, "phase": "load", "load_seconds": round(load_seconds, 2),
             "allocated_gb": round(weights_allocated, 3),
             "reserved_gb": round(weights_reserved, 3)})

    print(f"load: {load_seconds:.1f}s  allocated {weights_allocated:.2f}GB  "
          f"reserved {weights_reserved:.2f}GB")
    if estimated:
        print(f"      estimate was {estimated:.2f}GB "
              f"({estimated - weights_reserved:+.2f}GB vs reserved)\n")

    tokenizer = service._model._tokenizer

    # --- inference across prompt sizes ------------------------------------
    print(f"{'prompt':>8} {'gen':>5} {'latency':>9} {'tok/s':>7} "
          f"{'peak alloc':>11} {'peak resv':>10} {'overhead':>9}")

    for n_tokens in args.prompt_tokens:
        for repeat in range(args.repeats):
            prompt = resolve_prompt(
                config["prompt_template"], "You are a helpful assistant.", _make_prompt(tokenizer, n_tokens)
            )
            actual_prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            output = service._model._call(prompt, max_new_tokens=args.max_new_tokens)
            latency = time.perf_counter() - started

            generated = len(tokenizer(output, add_special_tokens=False).input_ids)
            peak_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
            peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            # Everything above the weights themselves: KV cache, activations,
            # logits. This is the number _OVERHEAD_GB is trying to approximate.
            overhead = peak_reserved - weights_reserved

            _record({**run, "phase": "inference", "repeat": repeat,
                     "requested_prompt_tokens": n_tokens,
                     "prompt_tokens": actual_prompt_tokens,
                     "generated_tokens": generated,
                     "latency_seconds": round(latency, 3),
                     "tokens_per_second": round(generated / latency, 2) if latency else None,
                     "peak_allocated_gb": round(peak_allocated, 3),
                     "peak_reserved_gb": round(peak_reserved, 3),
                     "overhead_gb": round(overhead, 3)})

            print(f"{actual_prompt_tokens:>8} {generated:>5} {latency:>8.2f}s "
                  f"{generated / latency if latency else 0:>7.1f} "
                  f"{peak_allocated:>10.2f}G {peak_reserved:>9.2f}G {overhead:>8.2f}G")

    print(f"\nappended to {RESULTS}")


if __name__ == "__main__":
    main()
