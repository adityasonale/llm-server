import os
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEYS: list[dict] = json.loads(os.getenv("API_KEYS", "[]"))
LOCAL_MODELS: dict = {
    "distilled-qwen-7b": {
        "path": "D:/vs code/python/models/LLMs/DeepSeek-R1-Distilled-Qwen-7B",
        "prompt_template": "qwen"
    },
    "minichat-1.5-3b": {
        "path": "D:/vs code/python/models/LLMs/MiniChat-1.5-3B",
        "prompt_template": "mistral"
    },
}
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")