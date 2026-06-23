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
        logger.info("Response received from Groq API")
        return response["choices"][0]["message"]["content"]