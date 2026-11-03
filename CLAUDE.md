# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A self-hosted FastAPI gateway for **local models**. There are no hosted providers (Groq was removed; it is in git history). Two endpoints:

- **`POST /v1/chat/completions`** — the main one. OpenAI-compatible: `messages`, `tools`, `stream` and sampling params, so any OpenAI client works (OpenAI SDK, LangChain `ChatOpenAI(base_url=...)`, curl). The provider is a prefix on `model`: `llamacpp/<model>` or `local/<model>` (see *OpenAI-compatible endpoint* below). Auth is `Authorization: Bearer <key>` or `X-API-Key`. Errors are OpenAI-shaped: `{"error": {"message", "type", "code"}}`.
- **`POST /generate`** — the original, kept for existing callers. Body `{system_prompt, user_prompt, model, provider}`, response `{content, model, provider}`, `X-API-Key` header only, no generation parameters. Only `provider: "local"` exists now.

`GET /health` is unauthenticated and returns `{"status": "ok"}`.

## Running

```bash
pip install -r requirements-local.txt   # core + the local provider's torch stack
uvicorn main:app --port 8000
```

Dependencies are split across two files: `requirements.txt` is the core server (FastAPI, httpx for the upstream pass-through, requests, langsmith) and `requirements-local.txt` includes it via `-r` and adds the local provider's stack (torch, transformers, bitsandbytes, accelerate, langchain-core, psutil). Both pin the CPU build of torch, since that is what PyPI serves; the CUDA build used in development is installed separately (see the comment in `requirements-local.txt`).

Note that `utils/system.py` imports `torch` unconditionally and is imported by `providers/local/services.py`, which `router.py` imports at module load. **Torch is therefore required to start the server even when only using llama-server** — which is why the Docker image installs the full stack. `local_llm_server.py` already defers its own `torch`/`transformers` imports into method bodies, so `utils/system.py` (plus `psutil`) is the only remaining eager edge. Deferring it is the prerequisite for a slim image; it has been tried and reverted once, so discuss before attempting it again.

Run from the repo root. `providers/` and `utils/` have no `__init__.py` (they resolve as namespace packages) and every import is absolute from the root, and `utils/logger.py` opens `Path("logs")` relative to the CWD — starting uvicorn from elsewhere breaks both.

`--reload` is available but costs you the resident local model on every save, since a reload is a fresh process. There is no test suite or linter config in the repo, and no test framework is installed — checks are written as stdlib `assert`s in a `__main__` block (see `providers/local/capacity.py`).

## Measurement

```bash
python -m providers.local.capacity          # self-check + size a model, no GPU needed
python benchmark.py --model minichat-1.5-3b # load time, latency, peak VRAM
```

`benchmark.py` appends one JSON object per measurement to `benchmarks/results.jsonl`, stamped with the git commit and GPU name. It is a deliberate, manually-run experiment, not monitoring — the point is comparing **the same model to itself** before and after a change (quantization, `context_length`, a dependency upgrade). Rows for different models in that file are separate runs, not a comparison.

Peak memory comes from `torch.cuda.max_memory_allocated()` with `reset_peak_memory_stats()` rather than sampling around the call, since the `log_memory()` calls in `LocalService` cannot see a spike that occurs *during* generation.

Two known flaws in the current harness: the `overhead_gb` column is computed from *reserved* memory, which the allocator keeps flat, so `peak_allocated_gb` minus the post-load `allocated_gb` is the honest signal; and generation length is not pinned, so a model that hits EOS early produces rows whose latencies are not comparable.

## Docker

```bash
docker compose up --build       # build and run on :8000
docker compose logs -f          # follow output
docker compose down             # stop
```

`docker-compose.yml` injects `.env` via `env_file` (secrets are never baked into the image), bind-mounts `./logs` so the rotating log survives the container, and keeps a named `hf-cache` volume at `/home/appuser/.cache/huggingface` so Hub downloads are not re-fetched on every recreate. The image runs as non-root `appuser` and healthchecks `/health` using stdlib `urllib`, as the slim base has no `curl`.

**Last verified** (before Groq was removed): the image built at **1.33GB**, the container reported healthy, and `POST /generate` returned 200 end to end through the then-Groq provider (auth, LangSmith tracing, token logging, and the `./logs` mount). Not re-verified since the switch to local-only; with no GPU in the container, the useful path there is `llamacpp/` pointed at a llama-server outside it.

The Dockerfile installs torch **from PyTorch's CPU index** (`--index-url https://download.pytorch.org/whl/cpu`) in its own layer, before the rest of the requirements. This is load-bearing: the default PyPI torch wheel bundles ~3GB of `nvidia-*` CUDA libraries that this container cannot use, and downloading them is what made the first build fail on a slow link. Keep it in a separate layer so a later failure does not re-fetch it, and keep the raised `--timeout`/`--retries`.

`provider: "local"` still does not work in the container, by design — there is no GPU passthrough, and 4-bit `BitsAndBytesConfig` expects CUDA, so a load would likely fail even after the weights resolve. Run local models on bare metal.

## Configuration

All config lives in `settings.py`, sourced from `.env` (gitignored):

- `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8089/v1`) and optional `LLAMACPP_API_KEY` — where the `llamacpp/` provider forwards to. Upstreams are the `OPENAI_COMPATIBLE_UPSTREAMS` dict; a vLLM server or a second llama-server is one more entry.
- `UPSTREAM_TIMEOUT_SECONDS` (default 300) — read timeout for upstream calls; connect timeout is fixed at 10s.
- `API_KEYS` — a **JSON array string** of `{"user", "project", "api_key"}` objects. This is the entire auth system; `authentication/api_key.py` flattens it into a dict at import time, so adding a key requires a server restart.
- `LANGCHAIN_*` — LangSmith tracing; `get_response` in `router.py` and non-streaming `chat_completion` in `chat.py` are decorated with `@traceable`, and both endpoints tag each trace with the API key's `user` and `project` via `langsmith_extra` metadata. Streaming requests are not traced.
- `LOG_LEVEL` (read directly in `utils/logger.py`, default `INFO`) — controls console verbosity only; the rotating file handler at `logs/llm-server.log` always logs DEBUG.

`LOCAL_MODELS` is hardcoded in `settings.py` and maps a model name to `path` (optional), `repo_id` (optional) and `prompt_template`. `_resolve_source()` in `providers/local/services.py` prefers `path` when that directory actually exists and otherwise falls back to downloading `repo_id` from the Hugging Face Hub, flipping `local_files_only` accordingly; configuring neither raises `ValueError`. The `path` values point outside the repo (`D:/vs code/python/models/LLMs/...`) and are machine-specific, which is exactly why the Hub fallback exists.

**The Hub fallback has never been exercised** — on the development machine the local paths always win, and triggering it means a 15GB download. To test it cheaply, add a tiny model (e.g. `hf-internal-testing/tiny-random-gpt2`) as a `repo_id`-only entry. Also note `MiniChat-1.5-3B`'s local `config.json` reports `_name_or_path: MiniChat-1.5-DPO-3B`, so the configured `repo_id` may not match the weights on disk.

## Architecture

Request flow for `/generate`: `main.py` (validate API key → validate body) → `router.py` (`PROVIDERS` registry lookup, timing, LangSmith trace) → a `LLMProvider` subclass.

Request flow for `/v1/chat/completions`: `openai_api.py` (auth dependency → `ChatCompletionRequest` → error mapping) → `chat.py` (`parse_model` → dispatch, usage logging, tracing) → either an `OpenAICompatibleProvider` or `LocalService.chat()`. See the next section.

**Adding a `/generate` provider:** implement `providers/base.py::LLMProvider.fetch_response(system_prompt, user_prompt, model) -> str` and register the instance in `PROVIDERS` in `router.py`. The dict is built at import time, so provider `__init__` must be cheap and must not load models eagerly. An unknown provider raises `ValueError`, which `main.py` maps to HTTP 400; anything else becomes a 500.

`fetch_response` is declared sync on the ABC but every call site awaits it — implementations must be `async def`.

**Local** (`providers/local/`): `CustomLLM` is a LangChain `LLM` subclass wrapping `transformers` with 4-bit quantization by default. `LocalService` holds **one model in memory at a time** and serializes everything behind an `asyncio.Lock` — see *Concurrency and admission control* below for how the resulting queue is bounded. `SystemMonitor` logs RAM/VRAM around each load, unload, and inference.

**Load order matters and is deliberate.** `_load_model` resolves and capacity-checks the incoming model *before* unloading the resident one. The obvious ordering — unload, then load — means a request that cannot fit destroys a working model and leaves nothing serving. Do not "simplify" it back.

`_check_capacity()` compares `capacity.py`'s estimate against available VRAM, and raises `ValueError` (→ HTTP 400) without touching the resident model when it will not fit. Since a resident model's memory returns on unload, it compares against `_available_vram_before_load` — the reading taken before the current model loaded — rather than what is free right now, otherwise every swap would be wrongly refused.

**VRAM must be measured as available-to-torch, not driver-free.** `torch.cuda.mem_get_info()` counts torch's reserved-but-unused cache as occupied. Measured on this machine: after unloading a 7B, the driver reported 4.04GB free while ~2.9GB more sat in torch's cache — and loading a 1.83GB model then moved driver usage by only 0.04GB, because it allocated straight out of that cache. `SystemMonitor.get_available_vram()` adds `reserved - allocated` back in; use it for any capacity decision. The post-unload leak warning uses the same measure, so if it fires now it means something real.

`capacity.py::estimate_vram_gb()` reads safetensors headers (a few KB) — never the weights — and returns `None` when it cannot size a model, so callers skip the check rather than acting on a wrong number. It is a pure function with no torch import, deliberately: it must stay testable without a GPU.

Measured accuracy: predicted weights were within 0.11GB (7B) and 0.04GB (3B) of actual. `_OVERHEAD_GB = 1.5` covers KV cache, activations and CUDA context; benchmarking showed overhead is **not** constant but grows ~0.0004GB per prompt token, reaching ~0.9GB near full context. A short-prompt measurement suggests 0.5 would do — that is the floor, not the ceiling, and lowering it will OOM on long prompts.

Two things about `CustomLLM` to keep in mind when changing this path: its generation settings (`max_new_tokens=2048`, `temperature=0.7`, `context_length=4096`, …) are constructor defaults that `/generate` never overrides (`/v1/chat/completions` can, per request, via `_call` kwargs), and `_call` catches generation failures and **returns** an error string instead of raising — so a failed local inference surfaces as HTTP 200 with the error text in `content`, not a 500.

Weight resolution is `_resolve_source()`'s job, not `CustomLLM`'s: it returns `(source, local_files_only)` and both are passed to the constructor, so `model_path` may be either a directory or a Hub repo id.

### Concurrency and admission control

Throughput is still **one inference at a time** — one GPU, one resident model, and the 7B fits with ~0.4GB to spare, so parallel generation would OOM. What is bounded is the waiting.

**Blocking work must stay off the event loop.** `_load_model` and `CustomLLM._call` are synchronous and take tens of seconds; `fetch_response` runs both through `asyncio.to_thread`. Calling them directly from the coroutine froze the entire process for the duration — `/health` stopped answering (failing the Docker healthcheck's 3s timeout), and requests to every other provider queued behind local ones. It also silently broke timeouts: an `asyncio` timer cannot fire while the loop is blocked, so any queueing built on top would have been decorative. Do not "simplify" the `to_thread` calls away.

`_acquire()` caps work in flight at `MAX_INFLIGHT_REQUESTS = 3` — **the running request counts**, so that is one running plus two waiting, and at ~33s per inference the last accepted waits ~66s. Derive the constant from the latency you will accept: `depth ≈ acceptable_wait × throughput`. Beyond the cap, requests are refused immediately rather than queued. `QUEUE_TIMEOUT_SECONDS = 120` bounds the wait for anything that is admitted.

Acquisition uses `asyncio.shield()` plus a done-callback: `wait_for` can otherwise cancel an acquire that has *already been granted*, leaking the lock permanently.

**The two failure modes are deliberately different HTTP codes:**

- **Saturation** → `ProviderOverloadedError` (`providers/base.py`) → **503** with `Retry-After`. Clears on its own, so retrying works.
- **Model too large for the device** → `ValueError` → **400**. Never clears — the weights do not fit this GPU at any point in the future — so a `Retry-After` would send clients into a futile loop. It sits alongside "unknown model" as the same class of unserveable request.

Verified with a stubbed model: the depth cap rejects in milliseconds, the timeout fires, the lock is released cleanly, and a 0.2s heartbeat keeps ticking through a blocking call that would previously have stopped it dead. **Not yet verified against a real CUDA generation** — the GIL is mostly released inside CUDA kernels, so the loop should stay free, but that is a prediction.

Next step if more throughput is ever needed: continuous batching (vLLM-style), where many requests share the GPU and admission is governed by KV cache capacity rather than a mutex.

Prompt formatting for local models is template-based (`providers/local/templates.py`): the `prompt_template` from `LOCAL_MODELS` selects one of `llama3`/`mistral`/`deepseek`/`qwen`. A new local model needs both an entry in `LOCAL_MODELS` and a matching template function.

### OpenAI-compatible endpoint

`parse_model()` in `chat.py` splits `model` on the **first** slash only: `llamacpp/Qwen/Qwen3-8B-GGUF` → provider `llamacpp`, model `Qwen/Qwen3-8B-GGUF`, because upstream model ids contain slashes. No prefix, or an unknown prefix, is a 400 listing the providers.

**`llamacpp/` is the main path for local models.** `providers/openai_compatible.py::OpenAICompatibleProvider` forwards the body **untouched apart from `model`** — `ChatCompletionRequest` declares only `model`/`messages`/`stream` and uses `extra="allow"`, so `tools`, `tool_choice`, `response_format`, `stream_options` and non-OpenAI params like `top_k` all reach llama-server. Tool calling and real token streaming are whatever the upstream supports. Responses (JSON and SSE bytes) are relayed as-is, so `model` in the response is the upstream's name, not the prefixed one. llm-server does **not** start or manage llama-server; run it separately on `LLAMACPP_BASE_URL`.

Upstream error mapping: upstream 4xx keeps its status and its OpenAI-shaped body is forwarded verbatim (with `Retry-After` on a 429); upstream 5xx → 502; unreachable → 502; timeout → 504. For `stream: true` the upstream request is sent **before** the `StreamingResponse` is returned, so an upstream error is still a real HTTP error rather than a 200 whose stream breaks off — keep that ordering. The `httpx.AsyncClient` per upstream is created lazily (provider `__init__` must stay cheap) and closed in `main.py`'s lifespan.

**`local/` is the in-process transformers path,** adapted rather than forwarded. `chat.py` reuses the **same `LocalService` instance as `/generate`** (`router.PROVIDERS["local"]`), so both endpoints share one GPU lock, one admission queue and one resident model — do not instantiate a second one. `LocalService.chat()` and `fetch_response()` both go through `_generate()`, so admission control and the load-order rules above apply unchanged. Messages are rendered by `templates.py::resolve_chat_prompt()` (multi-turn; `developer` → `system`; text parts flattened). For a single system + user pair it produces exactly what `resolve_prompt()` produces — `python -m providers.local.templates` asserts that. `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k` and `stop` map onto `CustomLLM._call` kwargs. Refused with 400 rather than silently dropped: `tools`/`functions`, `tool` messages or assistant `tool_calls`, non-text content, `n > 1`. `stream: true` is accepted but the whole answer arrives as **one** chunk, since generation is not incremental. No `usage` is returned.

Verified with the real OpenAI SDK and LangChain `ChatOpenAI` (invoke, `bind_tools`, stream) against a fake OpenAI-compatible upstream, and the `local/` path with a stubbed model. **Verified against a real llama-server** (PrismML llama.cpp fork, `Ternary-Bonsai-2-27B-PQ2_0.gguf`, `-c 8192 -ngl 99`, M4 MacBook Air 16GB): plain completion, streaming, and a LangChain `bind_tools` tool call all correct through llm-server. ~10.7 tok/s generation, ~20–50 tok/s prompt processing, ~5GB resident, no thermal warnings. llama-server returns the model's thinking in a separate `reasoning_content` field, so `content` stays clean. The `local/` path is still only verified with a stubbed model.

## Conventions

- Logging goes through `utils.logger.get_logger(__name__)`, which namespaces under `llm-server.*` with `propagate=False`. Use `%s` lazy formatting, not f-strings, in log calls.
- `logs/` is gitignored (it used to be tracked, which published a rejected API key in the log history). Never commit logs.
- **Never log a secret.** Log `authentication.api_key.fingerprint(key)` (SHA-256 prefix + length) instead — it identifies repeated attempts without revealing the key. `python -m authentication.api_key` self-checks it.
