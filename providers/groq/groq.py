import requests
from settings import GROQ_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_CHAT_API_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_STT_API_ENDPOINT = 'https://api.groq.com/openai/v1/audio/transcriptions'

class Groq:
    def __init__(self):
        super().__init__()
        self.api_key = GROQ_API_KEY

    def speech_to_text(self, audio_bytes: bytes, options: dict):
        if not audio_bytes:
            return {
                "success": False,
                "error": "Invalid audio data provided"
            }

        config = {
            "model": options.get("model", "whisper-large-v3"),
            "language": options.get("language", "en"),
            "temperature": options.get("temperature", 0),
            "response_format": options.get("response_format", "json"),
            "prompt": options.get("prompt", "")
        }

        files = {
            "file": ("audio.webm", audio_bytes, "audio/webm")
        }

        data = {
            "model": config["model"],
            "language": config["language"],
            "temperature": str(config["temperature"]),
            "response_format": config["response_format"]
        }

        if config["prompt"]:
            data["prompt"] = config["prompt"]

        try:
            response = requests.post(
                GROQ_STT_API_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                },
                files=files,
                data=data,
                timeout=60
            )

            if not response.ok:
                try:
                    err = response.json()
                    message = err.get("error", {}).get("message", response.text)
                except Exception:
                    message = response.text

                return {
                    "success": False,
                    "error": message
                }

            result = response.json()

            if "text" not in result:
                return {
                    "success": False,
                    "error": "No transcription text in API response"
                }

            return {
                "success": True,
                "text": result["text"],
                "language": result.get("language", config["language"]),
                "duration": result.get("duration")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}"
            }

    def fetch_response(self, system_prompt: str = "", user_prompt: str = "", config: dict = None):
        config = config or {}

        payload = {
            "model": config.get("model", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": config.get("temperature", 0.1),
            "max_tokens": config.get("max_tokens", 500)
        }

        try:
            response = requests.post(
                GROQ_CHAT_API_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=60
            )

            if not response.ok:
                raise RuntimeError(
                    f"Groq API error {response.status_code}: {response.text}"
                )

            return response.json()

        except Exception as e:
            logger.error("Groq API call failed: %s", e, exc_info=True)
            raise