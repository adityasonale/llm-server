import hashlib

from settings import API_KEYS
from utils.logger import get_logger

logger = get_logger(__name__)

# Build a lookup dict keyed by api_key for O(1) lookups
_KEY_MAP: dict[str, dict] = {entry["api_key"]: entry for entry in API_KEYS}


def fingerprint(secret: str | None) -> str:
    """
    A loggable stand-in for a secret: the first 8 hex chars of its SHA-256 and
    its length. Enough to tell repeated attempts with the same key apart, or to
    match against a known key's fingerprint — never enough to recover the key.
    Never log a key itself.
    """
    if not secret:
        return "<empty>"
    digest = hashlib.sha256(secret.encode()).hexdigest()[:8]
    return f"sha256:{digest} len={len(secret)}"


def validate_api_key(api_key: str) -> dict | None:
    """
    Returns the matching entry {user, project, api_key} if valid, None otherwise.
    """
    entry = _KEY_MAP.get(api_key)

    if entry is None:
        logger.warning("Invalid API key attempt: %s", fingerprint(api_key))
        return None

    logger.info("Authorised request — user: %s, project: %s", entry["user"], entry["project"])
    return entry


if __name__ == "__main__":
    # The raw key must never appear in a log line built from fingerprint().
    secret = "sk-live-0123456789abcdef"
    fp = fingerprint(secret)
    assert secret not in fp and "0123456789" not in fp, fp
    assert fp == fingerprint(secret) and fp != fingerprint(secret + "x")
    assert fingerprint("") == fingerprint(None) == "<empty>"
    print("fingerprint self-check passed:", fp)
