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
- ⬜ Stage 2 — push image to Artifact Registry
- ⬜ Stage 3 — Cloud Run GPU deploy: scale-to-zero, cold starts
- ⬜ Stage 4 — weight-loading strategy (GCS volume cache vs baked-in)

## Evaluation (the core LLMOps gap — not started)
The missing "did this change make it *better*?" signal. Everything else only answers "is it still running?".
- ⬜ Eval dataset — a small set of prompts with expected properties, versioned in the repo
- ⬜ Eval harness — run the dataset through `/generate`, score each output, record per commit
- ⬜ Scorers — start simple: exact-match, contains-keyword, JSON-valid, length bounds
- ⬜ LLM-as-judge — use the Groq provider to grade the local model's output (correctness, relevance)
- ⬜ Per-commit result storage (like the benchmark: one row per run, stamped with commit)
- ⬜ Regression tests on outputs — fail if quality drops vs the last baseline
- ⬜ Compare across axes — model (3B vs 7B), quantization (4-bit vs 8-bit), prompt A vs B
- ⬜ Prompt / config versioning so a scored result maps to exactly what produced it
- ⬜ (later) LangSmith datasets + evals integration

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

## Ordering note
Focus is MLOps/LLMOps; RAG and fine-tuning are out of scope for this project.
Weakest area (and highest priority) is **deployment + infra**, so the Cloud GPU
deploy track is worth finishing hands-on rather than rushing. Then pivot to
**evals + CI + registry** — the parts that make this LLMOps rather than plain
backend work. Hard rule: verify concurrency (task #1) before deploying to a
paid GPU, and build evals before CI (CI runs them).
