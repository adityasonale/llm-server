# LLM Server

A self-hosted FastAPI gateway for local models. It speaks the OpenAI chat-completions API, so any OpenAI client (OpenAI SDK, LangChain `ChatOpenAI`, curl) can use it.

## Models

Register models by name and path in `models.yaml` (copy `models.example.yaml`; it is gitignored because paths differ per machine):

```yaml
backends:
  llamacpp:
    binary: ~/llama.cpp/build/bin/llama-server
models:
  qwen3.5-9b:
    path: ~/models/Qwen3.5-9B-Q4_K_M.gguf
    args: ["-c", "16384"]          # optional backend args
    defaults: {temperature: 0.7}   # optional sampling defaults
```

Then ask for `model="qwen3.5-9b"`. llm-server detects the format from the path, picks a backend plugin for it on this machine, starts it on the first request, and keeps one model loaded at a time:

| Format | Backend plugin | Needs |
|---|---|---|
| `.gguf` file (or a folder with one) | `llamacpp` (llama-server) | llama-server binary |
| MLX folder | `mlx` (mlx_lm.server) | Apple Silicon, `pip install mlx-lm` |
| Hugging Face folder | `mlx`, else `vllm` | as above / Linux + NVIDIA + `vllm` |

`GET /v1/models` shows what each entry resolves to. New backends are `backends.base.Backend` subclasses, either built in (`backends/plugins.py`) or listed under `plugins:` in `models.yaml`.

## Legacy prefix

`llamacpp/<model>` is still accepted on `model`: it uses the managed model when `<model>` is in `models.yaml`, and otherwise forwards untouched to a llama-server you run yourself at `LLAMACPP_BASE_URL`.

## Usage

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<key from API_KEYS>")
reply = client.chat.completions.create(
    model="qwen3.5-9b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

For the unmanaged `llamacpp/` path, run llama-server yourself at `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8089/v1`).

llm-server itself has no GPU dependencies: it is a FastAPI gateway, and the models run in the backend processes it starts.
