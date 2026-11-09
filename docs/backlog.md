# llm-server — Backlog

Open work, in three lists: fixes for things that are wrong, new features, and upgrades to systems
that already work. Background: the ecosystem site (`~/projects/ecosystem`) and `docs/history.md`.
When an item is done, delete it here and record it in `docs/history.md`.

---

## Pending fixes

- **Docker image not re-verified** since the torch path was removed (2026-11-09). The image has no
  backend binaries or GPU, so inside it only the `llamacpp/` pass-through to an outside
  llama-server is useful. Rebuild and check it.
- **Hard kill leaves the backend running.** Ctrl+C stops backends through the lifespan, but a
  hard kill of llm-server orphans the llama-server process.
- **Streaming requests aren't traced** in LangSmith; only non-streaming `chat_completion` is.

## New features

- **Per-call prompt metrics.** Log prompt tokens, cached tokens and prompt time per call, tagged
  by the kind of call (router, supervisor, conversation…), to measure caching. Requested by Karen.
- **Generic thinking switch.** Translate a model-agnostic thinking on/off into each model's own
  flag in `models.yaml` (e.g. Qwen's `chat_template_kwargs.enable_thinking`), so clients stay
  model-agnostic.
- **Add API keys without a restart.** `API_KEYS` is read once at import time.

## Upgrades to existing systems

- **MLX plugin unverified.** `mlx-lm` wasn't installed when it was written; test it on the Mac.
- **vLLM plugin unverified.** Needs a Linux machine with an NVIDIA GPU.
- **Memory budget is coarse** (file size + 1 GB against a fraction of VRAM/RAM). Fine for refusing
  obvious misfits; revisit if it refuses models that would fit, or lets through ones that don't.
- **Faster backend on the Windows laptop.** Qwen3.5-9B runs on Docker Desktop's bundled Vulkan
  llama-server; an official llama.cpp CUDA build would likely be faster.
