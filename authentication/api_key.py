from settings import API_KEYS
from utils.logger import get_logger

logger = get_logger(__name__)

# Build a lookup dict keyed by api_key for O(1) lookups
_KEY_MAP: dict[str, dict] = {entry["api_key"]: entry for entry in API_KEYS}

def validate_api_key(api_key: str) -> dict | None:
    """
    Returns the matching entry {user, project, api_key} if valid, None otherwise.
    """
    entry = _KEY_MAP.get(api_key)

    if entry is None:
        logger.warning("Invalid API key attempt: %s", api_key)
        return None

    logger.info("Authorised request — user: %s, project: %s", entry["user"], entry["project"])
    return entry