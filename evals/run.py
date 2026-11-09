"""
Eval harness — runs a dataset of prompts through a provider, scores each output,
and records results per commit. Answers "is it correct?".

Run from the repo root:
    python -m evals.run                       # qwen3.5-9b
    python -m evals.run --model <name>        # any models.yaml entry

Calls the model manager in-process (no running server needed): it starts the
model's backend (e.g. llama-server) itself and stops it when the run ends.
"""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from chat import chat_completion, close_upstreams
from evals.scorers import score
from utils.logger import get_logger
from registry.registry import ModelRegistry

logger = get_logger(__name__)

# All *.jsonl files under dataset/ are loaded and run together.
DATASET_DIR = Path("evals/dataset")
# Each run writes its own timestamped file here, rather than appending to a
# shared log — so runs stay separate and easy to read/compare/delete.
RESULTS_DIR = Path("evals/results")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("LLM Evaluation")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _load_cases() -> list[dict]:
    cases = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    case = json.loads(line)
                    case.setdefault("dataset", path.stem)
                    cases.append(case)
    return cases


async def _ask(model: str, system: str, prompt: str) -> str:
    response = await chat_completion({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    })
    return response["choices"][0]["message"].get("content") or ""


async def main() -> None:
    try:
        await _run()
    finally:
        await close_upstreams()  # stops the backend the model manager started


async def _run() -> None:
    with mlflow.start_run() as run:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model", default="qwen3.5-9b")
        parser.add_argument("--benchmark-run-id", default=None)
        args = parser.parse_args()
        args.provider = "managed"  # kept in run metadata and file names

        cases = _load_cases()
        now = datetime.now(timezone.utc)
        run_meta = {
            "run_at": now.isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "host": platform.node(),
            "provider": args.provider,
            "model": args.model,
        }

        print(f"\n{args.provider}/{args.model} — {len(cases)} cases\n")
        print(f"{'id':<26} {'result':<6} {'detail'}")

        rows = []
        passed = 0
        for case in cases:
            started = time.perf_counter()
            try:
                output = await _ask(args.model, case["system"], case["prompt"])
                ok, detail = score(case["scorer"], output, case.get("expected"))
            except Exception as e:
                output, ok, detail = "", False, f"error: {e}"
            latency = time.perf_counter() - started

            passed += ok
            mark = "PASS" if ok else "FAIL"
            print(f"{case['id']:<26} {mark:<6} {detail}")
            if not ok:
                print(f"{'':<26}        output: {output[:80]!r}")

            rows.append({"case_id": case["id"], "dataset": case.get("dataset"),
                        "scorer": case["scorer"], "passed": bool(ok),
                        "latency_seconds": round(latency, 3), "output": output})

        total = len(cases)
        run_meta["passed"] = passed
        run_meta["total"] = total

        # One file per run: <timestamp>_<provider>_<model>.jsonl. First line is the
        # run summary, then one line per case.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%S")
        safe_model = args.model.replace("/", "-")
        out_path = RESULTS_DIR / f"{stamp}_{args.provider}_{safe_model}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"summary": run_meta}) + "\n")
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        mlflow.set_tag("mlflow.runName", f"{args.provider}-{args.model}-{now.strftime('%Y%m%d-%H%M')}")
        mlflow.set_tag("git_commit", run_meta["commit"])
        mlflow.set_tag("host", run_meta["host"])

        mlflow.log_param("provider", args.provider)
        mlflow.log_param("model", args.model)

        # after the results loop
        mlflow.log_metric("pass_rate", passed / total)
        mlflow.log_metric("passed", passed)
        mlflow.log_metric("total", total)
        mlflow.log_metric("avg_latency", sum(r["latency_seconds"] for r in rows) / total)

        mlflow.log_artifact(str(out_path))

        if passed / total >= 0.85:
            if not args.benchmark_run_id:
                print("Skipping registration — no --benchmark-run-id given.")
            else:
                registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")
                version = registry.register(
                    model_name=f"{args.provider}-{args.model}",
                    python_model=None,
                    benchmark_run_id=None,
                    eval_run_id=run.info.run_id,
                    active_run_id=run.info.run_id,
                    metadata={
                        "provider": args.provider,
                        "model": args.model,
                        "pass_rate": round(passed / total, 3),
                    }
                )
                print(f"Model registered as v{version} (eval passed {passed}/{total})")
        else:
            print(f"Eval score below threshold ({passed}/{total}), skipping registration.")
    
        print(f"\nscore: {passed}/{total} ({100 * passed / total:.0f}%)  -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
