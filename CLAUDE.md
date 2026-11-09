# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted FastAPI gateway for **local models**. llm-server never loads a model itself: each model runs in a separate backend process (llama-server, vLLM, mlx_lm.server) that llm-server starts, health-checks and forwards to. There are no hosted providers (Groq was removed), and the old in-process torch/transformers path (`/generate`, the `local/` prefix, `LocalService`, `LOCAL_MODELS`, `benchmark.py`) was removed too — both are in git history.

- **`POST /v1/chat/completions`** — OpenAI-compatible: `messages`, `tools`, `stream` and sampling params, so any OpenAI client works (OpenAI SDK, LangChain `ChatOpenAI(base_url=...)`, curl). `model` is a name from `models.yaml`, served by a backend plugin (see *Model manager and backend plugins*). The older `llamacpp/<model>` prefix still works (see *OpenAI-compatible endpoint*). Auth is `Authorization: Bearer <key>` or `X-API-Key`. Errors are OpenAI-shaped: `{"error": {"message", "type", "code"}}`.
- **`GET /v1/models`** (authenticated) lists the `models.yaml` entries with each one's detected format, the backend it would use on this machine, and whether it is loaded.
- **`GET /health`** is unauthenticated and returns `{"status": "ok"}`.

## Running

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

`requirements.txt` is the whole server: FastAPI, httpx, PyYAML, psutil, langsmith. No torch. Backends are installed separately (a llama-server binary; `mlx-lm` or `vllm` in whichever Python runs them).

Run from the repo root. `providers/` and `utils/` have no `__init__.py` (they resolve as namespace packages) and every import is absolute from the root, and `utils/logger.py` opens `Path("logs")` relative to the CWD — starting uvicorn from elsewhere breaks both.

`--reload` is available but restarts the resident backend on every save. There is no test suite or linter config, and no test framework is installed — checks are written as stdlib `assert`s in a `__main__` block:

```bash
python -m model_manager                     # self-check: formats, plugin selection, swaps (no models needed)
python -m authentication.api_key            # self-check: key fingerprinting
python -m evals.run --model qwen3.5-9b      # eval dataset against a models.yaml entry (needs mlflow)
```

`evals/run.py` calls `chat.chat_completion` in-process, so it starts the model's backend through the model manager without a running server, and stops it at the end. Results go to `evals/results/` and MLflow (`mlflow.db`).

## Docker

```bash
docker compose up --build       # build and run on :8000
docker compose logs -f          # follow output
docker compose down             # stop
```

`docker-compose.yml` injects `.env` via `env_file` (secrets are never baked into the image) and bind-mounts `./logs` so the rotating log survives the container. The image runs as non-root `appuser` and healthchecks `/health` using stdlib `urllib`, as the slim base has no `curl`. The image has no backend binaries and no GPU, so inside it the useful path is `llamacpp/` pointed at a llama-server outside the container. Not re-verified since the torch path was removed.

## Configuration

All config lives in `settings.py`, sourced from `.env` (gitignored):

- `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8089/v1`) and optional `LLAMACPP_API_KEY` — where the unmanaged `llamacpp/` prefix forwards to. Upstreams are the `OPENAI_COMPATIBLE_UPSTREAMS` dict; a long-running vLLM server or a second llama-server is one more entry.
- `UPSTREAM_TIMEOUT_SECONDS` (default 300) — read timeout for upstream calls; connect timeout is fixed at 10s.
- `API_KEYS` — a **JSON array string** of `{"user", "project", "api_key"}` objects. This is the entire auth system; `authentication/api_key.py` flattens it into a dict at import time, so adding a key requires a server restart.
- `LANGCHAIN_*` — LangSmith tracing; non-streaming `chat_completion` in `chat.py` is decorated with `@traceable` and tagged with the API key's `user` and `project` via `langsmith_extra` metadata. Streaming requests are not traced.
- `LOG_LEVEL` (read directly in `utils/logger.py`, default `INFO`) — controls console verbosity only; the rotating file handler at `logs/llm-server.log` always logs DEBUG.

`models.yaml` (gitignored, per machine; template in `models.example.yaml`; path overridable with `MODELS_FILE`) is the model registry: `models.<name>.path` plus optional `args`, `defaults`, `backend`, `port`, `startup_timeout` and plugin-specific keys; `backends.<plugin>` holds machine settings such as the llama-server `binary`; `plugins:` lists extra plugins as `module:Class`. It is re-read whenever its mtime changes, so adding a model needs no restart.

## Architecture

Request flow: `openai_api.py` (auth dependency → `ChatCompletionRequest` → error mapping) → `chat.py` (`parse_model` → dispatch, usage logging, tracing) → either the model manager's resident backend or an unmanaged `OpenAICompatibleProvider` upstream.

### Model manager and backend plugins

`model_manager.py::MANAGER` serves `models.yaml` names. Per request it runs `backends/formats.py::detect_format(path)` (`gguf` = a .gguf file or a folder holding one, split files via the first shard, `mmproj*` ignored; `mlx` = a folder whose `config.json` has a top-level `quantization` with `bits`; `hf` = `config.json` + `*.safetensors`/`*.bin`), then `backends/plugins.py::select_backend()`: a pinned `backend:` wins (and must suit the format and hardware, or it is a 400); otherwise the highest-`priority` plugin that handles the format and whose `unavailable_reason(hw, settings)` is `None`. Built-ins: `llamacpp` 100 (gguf), `mlx` 90 (mlx, hf; Apple Silicon + `mlx-lm`), `vllm` 80 (hf; Linux + NVIDIA + `vllm`). There is no in-process fallback: an `hf` model on a machine with neither mlx nor vllm is a 400. `backends/hardware.py` detects Apple Silicon / NVIDIA VRAM (via `nvidia-smi`) / RAM (via `psutil`) once.

A plugin is a `backends/base.py::Backend` subclass; a fresh instance per load holds that load's state. Server plugins subclass `backends/server.py::ServerBackend`, which launches the process with `subprocess.Popen` (not asyncio subprocesses: those need the Proactor loop on Windows, and the NVIDIA laptop is Windows), logs to `logs/backends/<model>.log`, polls `health_path` until 200, then forwards through an `OpenAICompatibleProvider`, so tools, streaming and error mapping are the same as the `llamacpp/` pass-through. A process that exits during load → `UpstreamError` 502 with the log tail; never healthy within `startup_timeout` → 504.

**One resident model.** `acquire(name)` returns the running model and counts it in use; every acquire is paired with `release()` (for streams, in the relay's `finally`, so a swap cannot stop a model mid-answer). Requests for the resident model run concurrently (the backend queues them). A different model sets `_swapping`, which holds off new requests, waits up to `DRAIN_TIMEOUT_SECONDS = 120` for in-flight ones (→ `ProviderOverloadedError` 503), then loads outside the condition lock.

**Load order matters and is deliberate.** `estimate_gb` vs `budget_gb` (coarse: file size + 1GB vs 75% of unified memory on Apple Silicon, VRAM + half of RAM for llama.cpp on CUDA, VRAM for vLLM) is checked **before** the resident model is stopped → `ValueError` 400. The obvious ordering — stop, then check — means a request that cannot fit destroys a working model and leaves nothing serving. A failed start leaves nothing resident; the next request retries. An entry whose `models.yaml` values change is restarted on its next request (`ModelSpec` equality).

Request bodies get the entry's `defaults` filled in under the client's values; `ServerBackend` replaces `model` with what the server expects (`default_model` for mlx_lm.server, which otherwise loads whatever `model` names). Server backends live in their own process group on POSIX, so Ctrl+C on uvicorn reaches only llm-server, whose lifespan (`close_upstreams()` → `MANAGER.shutdown()`) stops them. A hard kill of llm-server leaves the backend running.

**The two failure modes are deliberately different HTTP codes:**

- **Saturation** → `ProviderOverloadedError` (`providers/base.py`) → **503** with `Retry-After`. Clears on its own, so retrying works.
- **Model too large for the device** → `ValueError` → **400**. Never clears, so a `Retry-After` would send clients into a futile loop. It sits alongside "unknown model" as the same class of unserveable request.

**Verified** on the M4 Air with real Bonsai (`bonsai-27b` entry, PrismML llama-server): `/v1/models`, cold start on the first request (~6s to healthy), plain completion, a tool call via the `llamacpp/bonsai-27b` alias, streaming, unknown-name 400, and llama-server stopped on shutdown. `python -m model_manager` covers swaps, drain, the size refusal and failed starts with a fake plugin. **Not verified:** the MLX plugin (`mlx-lm` not installed) and the vLLM plugin (no CUDA Linux machine).

### OpenAI-compatible endpoint

`parse_model()` in `chat.py` first checks `models.yaml`: an exact name, or `llamacpp/<name>` where `<name>` is registered (an alias so clients configured with the old prefix need no change), goes to the model manager. Otherwise it splits `model` on the **first** slash only: `llamacpp/Qwen/Qwen3-8B-GGUF` → provider `llamacpp`, model `Qwen/Qwen3-8B-GGUF`, because upstream model ids contain slashes. No prefix, or an unknown prefix, is a 400 listing the providers.

`providers/openai_compatible.py::OpenAICompatibleProvider` forwards the body **untouched apart from `model`** — `ChatCompletionRequest` declares only `model`/`messages`/`stream` and uses `extra="allow"`, so `tools`, `tool_choice`, `response_format`, `stream_options`, `chat_template_kwargs` and non-OpenAI params like `top_k` all reach the backend. Tool calling and real token streaming are whatever the backend supports. Responses (JSON and SSE bytes) are relayed as-is, so `model` in the response is the backend's name. The unmanaged `llamacpp/` path does **not** start llama-server; run it separately on `LLAMACPP_BASE_URL`.

Upstream error mapping: upstream 4xx keeps its status and its OpenAI-shaped body is forwarded verbatim (with `Retry-After` on a 429); upstream 5xx → 502; unreachable → 502; timeout → 504. For `stream: true` the upstream request is sent **before** the `StreamingResponse` is returned, so an upstream error is still a real HTTP error rather than a 200 whose stream breaks off — keep that ordering. The `httpx.AsyncClient` per upstream is created lazily (provider `__init__` must stay cheap) and closed in `main.py`'s lifespan.

**Blocking work must stay off the event loop.** Anything slow and synchronous (process waits, file reads of model headers) goes through `asyncio.to_thread`, as `ServerBackend.stop()` does. Blocking the loop stops `/health` answering and silently breaks every `asyncio` timeout built on top.

Verified with the real OpenAI SDK and LangChain `ChatOpenAI` (invoke, `bind_tools`, stream) against a fake OpenAI-compatible upstream, and **against a real llama-server** (PrismML llama.cpp fork, `Ternary-Bonsai-2-27B-PQ2_0.gguf`, `-c 8192 -ngl 99`, M4 MacBook Air 16GB): plain completion, streaming, and a LangChain `bind_tools` tool call all correct through llm-server. llama-server returns the model's thinking in a separate `reasoning_content` field, so `content` stays clean.

## Conventions

- Logging goes through `utils.logger.get_logger(__name__)`, which namespaces under `llm-server.*` with `propagate=False`. Use `%s` lazy formatting, not f-strings, in log calls.
- `logs/` is gitignored (it used to be tracked, which published a rejected API key in the log history). Never commit logs.
- **Never log a secret.** Log `authentication.api_key.fingerprint(key)` (SHA-256 prefix + length) instead — it identifies repeated attempts without revealing the key. `python -m authentication.api_key` self-checks it.
