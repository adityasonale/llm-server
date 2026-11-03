import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEYS: list[dict] = json.loads(os.getenv("API_KEYS", "[]"))
# Where model weights live on disk. Unset locally -> the machine-specific dev
# paths below are used. Set in the container (e.g. to a mounted GCS bucket) ->
# each model is looked up under MODELS_DIR/<subdir>, avoiding a Hub download.
MODELS_DIR = os.getenv("MODELS_DIR")


def _model_path(dev_path: str, subdir: str) -> str:
    return os.path.join(MODELS_DIR, subdir) if MODELS_DIR else dev_path


# "path" is optional: if it exists on disk the weights are loaded from there,
# otherwise "repo_id" is downloaded from the Hugging Face Hub. At least one of
# the two must be set.
LOCAL_MODELS: dict = {
    "distilled-qwen-7b": {
        "path": _model_path("D:/vs code/python/models/LLMs/DeepSeek-R1-Distilled-Qwen-7B", "DeepSeek-R1-Distilled-Qwen-7B"),
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prompt_template": "qwen"
    },
    "minichat-1.5-3b": {
        "path": _model_path("D:/vs code/python/models/LLMs/MiniChat-1.5-3B", "MiniChat-1.5-3B"),
        "repo_id": "GeneZC/MiniChat-1.5-3B",
        "prompt_template": "mistral"
    },
}
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
