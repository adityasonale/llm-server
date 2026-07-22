# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted FastAPI gateway that fronts multiple LLM providers behind a single `POST /generate` endpoint. Callers pass `provider` + `model` in the body and an `X-API-Key` header.

The contract is fixed by the pydantic models in `main.py`: the body is `{system_prompt, user_prompt, model, provider}` (all required strings), the response is `{content, model, provider}`. There is no way to pass generation parameters — temperature, max tokens, etc. are hardcoded per provider (see Architecture). `GET /health` is unauthenticated and returns `{"status": "ok"}`.

## Running

```bash
pip install -r requirements-local.txt   # core + the local provider's torch stack
uvicorn main:app --port 8000
```

Dependencies are split across two files: `requirements.txt` is the core server (FastAPI, requests, langsmith) and `requirements-local.txt` includes it via `-r` and adds the local provider's stack (torch, transformers, bitsandbytes, accelerate, langchain-core, psutil). Both pin the CPU build of torch, since that is what PyPI serves; the CUDA build used in development is installed separately (see the comment in `requirements-local.txt`).

Note that `utils/system.py` imports `torch` unconditionally and is imported by `providers/local/services.py`, which `server.py` imports at module load. **Torch is therefore required to start the server even when only using Groq** — which is why the Docker image installs the full stack. `local_llm_server.py` already defers its own `torch`/`transformers` imports into method bodies, so `utils/system.py` (plus `psutil`) is the only remaining eager edge. Deferring it is the prerequisite for a slim image; it has been tried and reverted once, so discuss before attempting it again.

Run from the repo root. `providers/` and `utils/` have no `__init__.py` (they resolve as namespace packages) and every import is absolute from the root, and `utils/logger.py` opens `Path("logs")` relative to the CWD — starting uvicorn from elsewhere breaks both.

`--reload` is available but costs you the resident local model on every save, since a reload is a fresh process. There is no test suite or linter config in the repo.

## Docker

```bash
docker compose up --build       # build and run on :8000
docker compose logs -f          # follow output
docker compose down             # stop
```

`docker-compose.yml` injects `.env` via `env_file` (secrets are never baked into the image), bind-mounts `./logs` so the rotating log survives the container, and keeps a named `hf-cache` volume at `/home/appuser/.cache/huggingface` so Hub downloads are not re-fetched on every recreate. The image runs as non-root `appuser` and healthchecks `/health` using stdlib `urllib`, as the slim base has no `curl`.

**Verified working:** the image builds at **1.33GB**, the container reports healthy, and `POST /generate` with `provider: "groq"` returns 200 end to end (auth, LangSmith tracing, token logging, and the `./logs` mount all confirmed).

The Dockerfile installs torch **from PyTorch's CPU index** (`--index-url https://download.pytorch.org/whl/cpu`) in its own layer, before the rest of the requirements. This is load-bearing: the default PyPI torch wheel bundles ~3GB of `nvidia-*` CUDA libraries that this container cannot use, and downloading them is what made the first build fail on a slow link. Keep it in a separate layer so a later failure does not re-fetch it, and keep the raised `--timeout`/`--retries`.

`provider: "local"` still does not work in the container, by design — there is no GPU passthrough, and 4-bit `BitsAndBytesConfig` expects CUDA, so a load would likely fail even after the weights resolve. Run local models on bare metal.

## Configuration

All config lives in `settings.py`, sourced from `.env` (gitignored):

- `GROQ_API_KEY`
- `API_KEYS` — a **JSON array string** of `{"user", "project", "api_key"}` objects. This is the entire auth system; `authentication/api_key.py` flattens it into a dict at import time, so adding a key requires a server restart.
- `LANGCHAIN_*` — LangSmith tracing; `get_response` in `server.py` is decorated with `@traceable`.
- `LOG_LEVEL` (read directly in `utils/logger.py`, default `INFO`) — controls console verbosity only; the rotating file handler at `logs/llm-server.log` always logs DEBUG.

`LOCAL_MODELS` is hardcoded in `settings.py` and maps a model name to `path` (optional), `repo_id` (optional) and `prompt_template`. `_resolve_source()` in `providers/local/services.py` prefers `path` when that directory actually exists and otherwise falls back to downloading `repo_id` from the Hugging Face Hub, flipping `local_files_only` accordingly; configuring neither raises `ValueError`. The `path` values point outside the repo (`D:/vs code/python/models/LLMs/...`) and are machine-specific, which is exactly why the Hub fallback exists.

**The Hub fallback has never been exercised** — on the development machine the local paths always win, and triggering it means a 15GB download. To test it cheaply, add a tiny model (e.g. `hf-internal-testing/tiny-random-gpt2`) as a `repo_id`-only entry. Also note `MiniChat-1.5-3B`'s local `config.json` reports `_name_or_path: MiniChat-1.5-DPO-3B`, so the configured `repo_id` may not match the weights on disk.

## Architecture

Request flow: `main.py` (validate API key → validate body) → `server.py` (`PROVIDERS` registry lookup, timing, LangSmith trace) → a `LLMProvider` subclass.

**Adding a provider:** implement `providers/base.py::LLMProvider.fetch_response(system_prompt, user_prompt, model) -> str` and register the instance in `PROVIDERS` in `server.py`. The dict is built at import time, so provider `__init__` must be cheap and must not load models eagerly. An unknown provider raises `ValueError`, which `main.py` maps to HTTP 400; anything else becomes a 500.

`fetch_response` is declared sync on the ABC but every call site awaits it — implementations must be `async def`.

**Groq** (`providers/groq/`): `groq.py` is a thin blocking `requests` wrapper returning the raw OpenAI-shaped JSON (it also has an unused `speech_to_text` method); `services.py` adapts it to the ABC, logs token usage, and unwraps `choices[0].message.content`. The blocking call runs directly on the event loop. `fetch_response` accepts a `config` dict, but `services.py` only ever passes `model` through — so `temperature=0.1` and `max_tokens=500` are effectively hardcoded, as is the `llama-3.3-70b-versatile` fallback model that a missing `config` would select.

**Local** (`providers/local/`): `CustomLLM` is a LangChain `LLM` subclass wrapping `transformers` with 4-bit quantization by default. `LocalService` holds **one model in memory at a time** — a request for a different model unloads the current one first — and serializes everything behind an `asyncio.Lock`, so concurrent requests queue. `SystemMonitor` logs RAM/VRAM around each load, unload, and inference.

Two things about `CustomLLM` to keep in mind when changing this path: its generation settings (`max_new_tokens=2048`, `temperature=0.7`, `context_length=4096`, …) are constructor defaults that nothing overrides, and `_call` catches generation failures and **returns** an error string instead of raising — so a failed local inference surfaces as HTTP 200 with the error text in `content`, not a 500.

Weight resolution is `_resolve_source()`'s job, not `CustomLLM`'s: it returns `(source, local_files_only)` and both are passed to the constructor, so `model_path` may be either a directory or a Hub repo id.

Prompt formatting for local models is template-based (`providers/local/templates.py`): the `prompt_template` from `LOCAL_MODELS` selects one of `llama3`/`mistral`/`deepseek`/`qwen`. A new local model needs both an entry in `LOCAL_MODELS` and a matching template function.

## Conventions

- Logging goes through `utils.logger.get_logger(__name__)`, which namespaces under `llm-server.*` with `propagate=False`. Use `%s` lazy formatting, not f-strings, in log calls.
- `logs/llm-server.log` is tracked in git and shows up as modified after any run.
