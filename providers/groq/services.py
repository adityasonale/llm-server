from .groq import Groq
from ..base import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

class GroqService(LLMProvider):

    def __init__(self):
        self.model = Groq()

    async def fetch_response(self, system_prompt:str, user_prompt:str, model: str):
        logger.info("Calling Groq API with model: %s", model)
        response = self.model.fetch_response(
            system_prompt,
            user_prompt,
            config={"model": model}
        )
        usage = response.get("usage", {})
        logger.info(
            "Groq API response received — model: %s, prompt_tokens: %s, completion_tokens: %s, total_tokens: %s",
            model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        return response["choices"][0]["message"]["content"]