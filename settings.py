import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEYS: list[dict] = json.loads(os.getenv("API_KEYS", "[]"))
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Local inference servers that already speak the OpenAI chat-completions API.
# /v1/chat/completions forwards to them untouched (tools, streaming, sampling
# params all pass through). Clients select one with a provider prefix on the
# model: "llamacpp/<anything>" (llama-server serves whatever GGUF it was started
# with). Adding vLLM or a second llama-server is one more entry here.
OPENAI_COMPATIBLE_UPSTREAMS: dict = {
    "llamacpp": {
        "base_url": os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8089/v1"),
        "api_key": os.getenv("LLAMACPP_API_KEY"),
    },
}
# Read timeout for upstream calls. Generous because a local model on a long
# completion can take minutes; connect timeout is fixed at 10s.
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300"))
