# llm-server — History

How llm-server got its current shape, 2–3 lines per change. Current design: the ecosystem site
(`~/projects/ecosystem`). Open work: `docs/backlog.md`.

- **2026-06-22: first server.** FastAPI LLM server with a local LLM service; LangSmith tracing
  added the next day.
- **2026-07-21: containerized.** Dockerfile and split requirements, the rotating logger, and a
  fallback to the Hugging Face Hub for local models.
- **2026-07-22: VRAM preflight + benchmarks.** Local models are checked against VRAM before
  loading; a benchmark harness was added (results from an earlier flawed harness were dropped).
- **2026-07-24: GPU serving.** A GPU path with bounded local concurrency; weights served from a
  mounted GCS bucket on Cloud Run.
- **2026-07-26 – 07-29: evals + MLflow.** Eval harness, MLflow tracking for benchmarks and evals,
  a model registry linked to both; LangSmith traces tagged with the key's user and project.
- **2026-11-03: OpenAI-compatible, local only.** `POST /v1/chat/completions` works with any OpenAI
  client; Groq was removed. Rejected API keys stopped being logged (fingerprints instead), and
  `logs/` stopped being tracked.
- **2026-11-06: models by name.** `models.yaml` entries served by backend plugins (llama.cpp, MLX,
  vLLM) chosen by format and hardware; one resident model, swaps drain in-flight requests.
- **2026-11-09: gateway only.** The in-process torch/transformers path (`/generate`, `local/`) was
  removed; every model runs in its own backend process. Today: Qwen3.5-9B on the RTX 4060.
- **2026-11-09: docs split.** `CLAUDE.md` keeps only instructions and rules; architecture is on
  the ecosystem site; pending work is in `docs/backlog.md`.
