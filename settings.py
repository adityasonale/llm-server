import os
import json

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEYS: list[dict] = json.loads(os.getenv("API_KEYS", "[]"))
