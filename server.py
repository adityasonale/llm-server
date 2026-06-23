from providers.base import LLMProvider
from providers.groq.services import GroqService
from providers.local.services import LocalService
from utils.logger import get_logger

logger = get_logger(__name__)

PROVIDERS: dict[str, LLMProvider] = {
    "groq": GroqService(),
    "local": LocalService(),
}

async def get_response(system_prompt: str, user_prompt: str, model: str, provider: str) -> str:
    logger.info("Routing request to provider: %s", provider)
    service = PROVIDERS.get(provider)

    if service is None:
        logger.error("Unknown provider requested: %s", provider)
        raise ValueError(f"Unknown provider: '{provider}'. Available: {list(PROVIDERS.keys())}")

    return await service.fetch_response(system_prompt, user_prompt, model)