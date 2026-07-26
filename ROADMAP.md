# LLMOps / MLOps Roadmap

Legend: ✅ done · 🔨 in progress · ⬜ planned

## Serving & infra
- ✅ Provider gateway (Groq + local), API-key auth, structured logging, LangSmith tracing
- ✅ Dockerized CPU/Groq image (1.33GB)
- ✅ VRAM preflight + capacity-aware model loading
- ✅ Concurrency / admission control — verified against a real model in the GPU container (responsive loop during load, queue timeout → 503)
- ✅ GPU image (`Dockerfile.cuda`) — builds and runs minichat on GPU in a container

## Cloud GPU deploy (current track)
- ✅ Stage 0 — GCP L4 quota approved (us-central1, no-zonal-redundancy)
- ✅ Stage 1 — GPU container runs minichat on GPU locally (weights cached to volume via Hub download)
- ✅ Stage 2 — image pushed to Artifact Registry (us-central1)
- ✅ Stage 3 — Cloud Run GPU deploy live, Groq path verified end-to-end (auth + serving)
- ✅ Stage 4 — weights served from a mounted GCS bucket (Cloud Storage FUSE), no per-cold-start download; local model verified end-to-end on Cloud Run GPU

## Evaluation (the core LLMOps gap)
The missing "did this change make it *better*?" signal. Everything else only answers "is it still running?".
- ✅ Eval dataset — `evals/dataset/*.jsonl`, versioned in the repo
- ✅ Eval harness — `evals/run.py` runs the dataset through a provider, scores each output
- ✅ Scorers — `evals/scorers.py`: contains, contains_ci, equals_ci, json_valid
- ✅ Per-run result storage — one timestamped file per run in `evals/results/`, stamped with commit
- ⬜ LLM-as-judge — use the Groq provider to grade open-ended output (correctness, relevance)
- ⬜ Regression gate — fail if score drops vs the last baseline (what CI runs)
- ⬜ Real dataset — swap hand-written cases for a GSM8K / TruthfulQA slice (same harness)
- ⬜ Compare across axes — model (3B vs 7B), quantization (4-bit vs 8-bit), prompt A vs B
- ⬜ Prompt / config versioning so a scored result maps to exactly what produced it
- ⬜ (later, end of roadmap) Learn one industry eval tool for interview fluency — **LangSmith** (already wired via `@traceable`; datasets + evals + LLM-judge in a UI). Promptfoo as a second name to know.

## Model registry & experiment tracking (MLOps core)
`LOCAL_MODELS` is already a primitive registry — formalize it.
- ⬜ Model registry — versions, source, quantization, VRAM cost, prompt template per model
- ⬜ Experiment tracking (MLflow or similar) — log each eval/benchmark run with params + metrics
- ⬜ Metric history per model — quality + latency + VRAM over time, comparable across commits
- ⬜ Promotion flow — mark a model/config "validated" before it's allowed to deploy
- ⬜ Link registry → eval results → deploy (only promoted configs ship)

## Automation & quality gates
- ⬜ CI on push — build image, run `capacity.py` self-check + eval, gate deploy on pass
- ⬜ Real test suite (currently just `assert`s in `__main__`)
- ⬜ Benchmark harness — fix the 2 known flaws, re-run a clean baseline

## Observability (production-grade)
- ⬜ Metrics endpoint — latency, tokens, VRAM, queue depth → Prometheus/Grafana
- ⬜ Output-quality monitoring, not just liveness
- ⬜ Cost tracking (tokens → $)

---

## Reference — production benchmarking & evaluation to build toward
A menu of what real systems measure, to implement one at a time in this repo.

### Performance benchmarks (speed / cost — extends `benchmark.py`)
- ⬜ TTFT — time to first token (needs streaming first)
- ⬜ TPOT — time per output token
- ⬜ Throughput — tokens/sec and requests/sec at a fixed concurrency
- ⬜ Load-test curve — latency vs concurrency, find the "knee" (`locust`/`k6`)
- ⬜ Cost per 1M tokens / GPU-hour
- Tools to try: `locust`, `k6`, vLLM benchmark scripts, NVIDIA `genai-perf`

### Quality evaluations (extends `evals/`)
Three tiers, increasing cost and fuzziness:
- ⬜ **Deterministic** — exact-match, regex, JSON-schema (have the basics)
- ⬜ **Model-graded** — LLM-as-judge for open-ended output; needs calibration
- ⬜ **Task benchmarks** — standard datasets:
  - GSM8K (math), HumanEval (code), MMLU (knowledge), TruthfulQA, MT-Bench (chat), RAGAS (RAG)
- Tools to try: LangSmith (already wired), Promptfoo, DeepEval, EleutherAI `lm-eval-harness`, RAGAS

### Pipelines (how these run in production)
- ⬜ **Offline (CI, pre-deploy):** commit → run eval set + benchmark → gate on `score ≥ baseline AND latency ≤ SLO` → allow/block deploy
- ⬜ **Online (post-deploy):** log live prompt/response/latency/cost → sample % → async LLM-judge → dashboards + alerts → feed failures back into the golden dataset
- ⬜ Golden/regression dataset — curated, versioned, grows from real failures (the key asset)
- ⬜ Baselines & thresholds — turn a score into a pass/fail gate
- ⬜ Canary / shadow deploy — new model serves small % of traffic, compare before rollout
- ⬜ Drift detection — input + output-quality shift over time

## Ordering note
Focus is MLOps/LLMOps; RAG and fine-tuning are out of scope for this project.
Weakest area (and highest priority) is **deployment + infra**, so the Cloud GPU
deploy track is worth finishing hands-on rather than rushing. Then pivot to
**evals + CI + registry** — the parts that make this LLMOps rather than plain
backend work. Hard rule: verify concurrency (task #1) before deploying to a
paid GPU, and build evals before CI (CI runs them).
