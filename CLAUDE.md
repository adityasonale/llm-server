# llm-server — Claude Working Notes

llm-server is a self-hosted FastAPI gateway for **local models only**. It never loads a model
itself: it starts, health-checks and forwards to one backend process per model (llama-server,
vLLM, mlx_lm.server) behind an OpenAI-compatible `POST /v1/chat/completions`.

**More detail:** the ecosystem site (`~/projects/ecosystem`, separate repo: `llm-server.html`,
`interfaces.html`, `decisions.html`) is the architecture reference. Also: `docs/history.md`
(what changed and why), `docs/backlog.md` (pending fixes, new features, upgrades), `README.md`,
`models.example.yaml` (the `models.yaml` format).

---

## How to work in this repo

These guidelines lean toward caution over speed. For trivial tasks, use judgment.

**1. Think before coding.** Don't assume, don't hide confusion, surface tradeoffs.
- State your assumptions. If you're unsure, ask.
- If there's more than one reading of the request, lay them out; don't pick one silently.
- If a simpler approach exists, say so. Push back when it's warranted.
- If something is unclear, stop, name what's confusing, and ask.

**2. Simplicity first.** Write the minimum code that solves the problem, nothing speculative.
- No features beyond what was asked, and no abstractions for code used once.
- No "flexibility" or configurability nobody asked for.
- No error handling for impossible cases.
- If you wrote 200 lines and 50 would do, rewrite it. Would a senior engineer call it overcomplicated?

**3. Surgical changes.** Touch only what you must; clean up only your own mess.
- Don't "improve" nearby code, comments or formatting, and don't refactor what isn't broken.
- Match the existing style, even if you'd do it differently.
- Unrelated dead code: mention it, don't delete it.
- Remove imports, variables and functions that *your* change made unused, nothing older.
- The test: every changed line traces back to the request.

**4. Goal-driven execution.** Define success criteria and loop until they're verified.
- There is no test framework: checks are stdlib `assert`s in a `__main__` block. Run
  `python -m model_manager` (formats, plugin choice, swaps, drain, size refusal; no models needed)
  and `python -m authentication.api_key`. Add asserts there for new behaviour.
- "Fix the bug" → reproduce it first (an assert or a script), then make it pass.
- For anything touching a backend, also check against a real model: plain completion, streaming,
  and a tool call (LangChain `bind_tools`).
- For multi-step work, state a short plan with a check per step: `1. [step] → verify: [check]`.

The guidelines are working if diffs carry fewer unneeded changes, fewer rewrites come from
overcomplication, and clarifying questions come before implementation, not after mistakes.

**Keeping the docs current:** new pending work goes in `docs/backlog.md`; when an item is done,
remove it there and add it to `docs/history.md` as one 2–3 line bullet (what changed and why,
no logs or test transcripts). Design changes go in the ecosystem pages.
This file holds only instructions and rules.

---

## Rules

**Gateway, not a model host**
- Local models only: no hosted providers, no in-process torch/transformers. Both were removed on
  purpose (they're in git history); don't bring them back without being asked.
- llm-server stays prompt-agnostic: it never builds or rewrites prompts. Clients own them.
- Forward request bodies untouched apart from `model` (`ChatCompletionRequest` uses
  `extra="allow"`), so `tools`, `chat_template_kwargs`, `top_k` and the rest reach the backend.
  Relay responses as-is.

**Model manager**
- One resident model. Every `acquire()` is paired with a `release()`; for streams, in the relay's
  `finally`, so a swap can't stop a model mid-answer.
- Check the memory budget **before** stopping the resident model. Stop-then-check lets a request
  that can't fit destroy a working model and leave nothing serving.
- Start backends with `subprocess.Popen`, not asyncio subprocesses: those need the Proactor loop
  on Windows, and the GPU laptop runs Windows.
- New backends are `Backend` / `ServerBackend` plugins chosen by format, hardware and priority.
  Don't add an in-process fallback.

**HTTP behaviour**
- Saturation → `ProviderOverloadedError` → **503** with `Retry-After` (it clears). A model too
  large for the device → `ValueError` → **400** with no `Retry-After` (it never clears). Keep
  these two distinct.
- Upstream 4xx keeps its status and OpenAI-shaped body; upstream 5xx or unreachable → 502;
  timeout → 504. Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`.
- For `stream: true`, send the upstream request **before** returning the `StreamingResponse`, so
  an upstream error is a real HTTP error, not a 200 whose stream breaks off. Keep that ordering.
- Keep blocking work off the event loop (`asyncio.to_thread`). A blocked loop stops `/health`
  answering and breaks every asyncio timeout.
- `parse_model()` splits on the **first** slash only: upstream model ids contain slashes.

**Secrets and logging**
- **Never log a secret.** Log `authentication.api_key.fingerprint(key)` instead.
- `API_KEYS` (in `.env`) is a JSON array of `{user, project, api_key}`; adding a key needs a restart.
- Log through `utils.logger.get_logger(__name__)` with `%s` lazy formatting, not f-strings.
  `logs/` is gitignored; never commit logs (they once published a rejected API key).

**Running**
- Run from the repo root: `providers/` and `utils/` are namespace packages with absolute imports,
  and the logger opens `logs/` relative to the CWD.
- Next to Karen, run on port **8100** (`uvicorn main:app --port 8100`); Karen's own API uses 8000.
- `models.yaml` is per machine and gitignored (template: `models.example.yaml`). It's re-read when
  it changes, so adding a model needs no restart. Model-specific flags belong there.
- `--reload` restarts the resident backend on every save.
- Config is `settings.py` from `.env`: `LLAMACPP_BASE_URL`, `UPSTREAM_TIMEOUT_SECONDS` (300),
  `API_KEYS`, `LANGCHAIN_*`, `MODELS_FILE`; `LOG_LEVEL` sets console verbosity only (the file log
  is always DEBUG).
- Docker: `docker compose up --build` injects `.env` via `env_file` (secrets never baked in);
  the image runs as non-root `appuser`.
- llama-server returns thinking in a separate `reasoning_content` field, so `content` stays clean.
