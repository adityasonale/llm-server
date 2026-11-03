# LLM Server

A self-hosted FastAPI gateway for local models. It speaks the OpenAI chat-completions API, so any OpenAI client (OpenAI SDK, LangChain `ChatOpenAI`, curl) can use it.

## Providers

Pick one with a prefix on `model`:

| Prefix | Backend | Tools | Streaming |
|---|---|---|---|
| `llamacpp/<model>` | llama.cpp `llama-server` (GGUF), forwarded untouched | yes | yes |
| `local/<model>` | in-process transformers, models from `LOCAL_MODELS` in `settings.py` | no | single chunk |

## Usage

```bash
pip install -r requirements-local.txt
uvicorn main:app --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<key from API_KEYS>")
reply = client.chat.completions.create(
    model="llamacpp/qwen3-8b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Run llama-server separately, at `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8089/v1`).

The original `POST /generate` endpoint (`{system_prompt, user_prompt, model, provider}`) still works for the `local` provider.
