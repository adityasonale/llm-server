# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted FastAPI gateway that fronts multiple LLM providers behind a single `POST /generate` endpoint. Callers pass `provider` + `model` in the body and an `X-API-Key` header.

## Running

```bash
pip install -r requirements-local.txt   # core + the local provider's torch stack
uvicorn main:app --port 8000
```

Dependencies are split across two files: `requirements.txt` is the core server (FastAPI, requests, langsmith) and `requirements-local.txt` includes it via `-r` and adds the local provider's stack (torch, transformers, bitsandbytes, accelerate, langchain-core, psutil). Both pin the CPU build of torch, since that is what PyPI serves; the CUDA build used in development is installed separately (see the comment in `requirements-local.txt`).

Note that `utils/system.py` imports `torch` unconditionally and is imported by `providers/local/services.py`, which `server.py` imports at module load. **Torch is therefore required to start the server even when only using Groq** — which is why the Docker image installs the full stack. Deferring those imports is the prerequisite for a slim image; it has been tried and reverted once, so discuss before attempting it again.

`--reload` is available but costs you the resident local model on every save, since a reload is a fresh process. There is no test suite or linter config in the repo.

## Docker

```bash
docker compose up --build       # build and run on :8000
docker compose logs -f          # follow output
docker compose down             # stop
```

`docker-compose.yml` injects `.env` via `env_file` (secrets are never baked into the image) and bind-mounts `./logs` so the rotating log survives the container. The image runs as non-root `appuser` and healthchecks `/health` using stdlib `urllib`, as the slim base has no `curl`.

Two caveats on the current image, both intentional and both worth fixing later:

- It is several GB, because the eager-import chain above forces the whole torch stack in.
- Torch inside it is the CPU build and no GPU passthrough is configured, so `provider: "local"` will technically run but at unusable speed. The container is for the Groq path; run local models on bare metal for now.

**The image has not been built or run yet** — these files are committed untested.

## Configuration

All config lives in `settings.py`, sourced from `.env` (gitignored):

- `GROQ_API_KEY`
- `API_KEYS` — a **JSON array string** of `{"user", "project", "api_key"}` objects. This is the entire auth system; `authentication/api_key.py` flattens it into a dict at import time, so adding a key requires a server restart.
- `LANGCHAIN_*` — LangSmith tracing; `get_response` in `server.py` is decorated with `@traceable`.
- `LOG_LEVEL` (read directly in `utils/logger.py`, default `INFO`) — controls console verbosity only; the rotating file handler at `logs/llm-server.log` always logs DEBUG.

`LOCAL_MODELS` is hardcoded in `settings.py` and maps a model name to an **absolute filesystem path** plus a `prompt_template` key. These paths point outside the repo (`D:/vs code/python/models/LLMs/...`) and are machine-specific.

## Architecture

Request flow: `main.py` (validate API key → validate body) → `server.py` (`PROVIDERS` registry lookup, timing, LangSmith trace) → a `LLMProvider` subclass.

**Adding a provider:** implement `providers/base.py::LLMProvider.fetch_response(system_prompt, user_prompt, model) -> str` and register the instance in `PROVIDERS` in `server.py`. The dict is built at import time, so provider `__init__` must be cheap and must not load models eagerly. An unknown provider raises `ValueError`, which `main.py` maps to HTTP 400; anything else becomes a 500.

`fetch_response` is declared sync on the ABC but every call site awaits it — implementations must be `async def`.

**Groq** (`providers/groq/`): `groq.py` is a thin blocking `requests` wrapper returning the raw OpenAI-shaped JSON (it also has an unused `speech_to_text` method); `services.py` adapts it to the ABC, logs token usage, and unwraps `choices[0].message.content`. The blocking call runs directly on the event loop.

**Local** (`providers/local/`): `CustomLLM` is a LangChain `LLM` subclass wrapping `transformers` with 4-bit quantization by default. `LocalService` holds **one model in memory at a time** — a request for a different model unloads the current one first — and serializes everything behind an `asyncio.Lock`, so concurrent requests queue. `SystemMonitor` logs RAM/VRAM around each load, unload, and inference.

Prompt formatting for local models is template-based (`providers/local/templates.py`): the `prompt_template` from `LOCAL_MODELS` selects one of `llama3`/`mistral`/`deepseek`/`qwen`. A new local model needs both an entry in `LOCAL_MODELS` and a matching template function.

## Conventions

- Logging goes through `utils.logger.get_logger(__name__)`, which namespaces under `llm-server.*` with `propagate=False`. Use `%s` lazy formatting, not f-strings, in log calls.
- `logs/llm-server.log` is tracked in git and shows up as modified after any run.
