# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted FastAPI gateway that fronts multiple LLM providers behind a single `POST /generate` endpoint. Callers pass `provider` + `model` in the body and an `X-API-Key` header.

## Running

```bash
pip install -r requirements.txt
uvicorn main:app --reload           # dev server on :8000
```

There is no test suite or linter config in the repo. `requirements.txt` pins the CPU build of torch; the CUDA build used in development is installed separately (see the comment in that file).

Note that `utils/system.py` imports `torch` unconditionally and is imported by `providers/local/services.py`, which `server.py` imports at module load. Torch is therefore required to start the server even when only using Groq.

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
