"""
Centralised logging for LLM Server.

Usage in any module:
    from utils.logger import get_logger
    logger = get_logger(__name__)

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
Console output is coloured; file output is plain-text with full timestamps.
Logs are written to logs/llm-server.log with 5 MB rotation and 5 backup files.
"""

import logging
import logging.handlers
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}

_LOG_DIR = Path("logs")

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _ColoredFormatter(logging.Formatter):
    """Adds ANSI colour to the level name in console output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


_CONSOLE_FMT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_FILE_FMT     = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_CONSOLE_DATE = "%H:%M:%S"
_FILE_DATE    = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Root "llm-server" logger — set up once
# ---------------------------------------------------------------------------
_initialised = False


def _setup() -> None:
    global _initialised
    if _initialised:
        return

    _LOG_DIR.mkdir(exist_ok=True)

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, log_level_str, logging.INFO)

    root = logging.getLogger("llm-server")
    root.setLevel(logging.DEBUG)   # handlers filter individually
    root.propagate = False

    # --- Console handler ---
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(_ColoredFormatter(fmt=_CONSOLE_FMT, datefmt=_CONSOLE_DATE))
    root.addHandler(console)

    # --- Rotating file handler ---
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "llm-server.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=_FILE_FMT, datefmt=_FILE_DATE))
    root.addHandler(file_handler)

    _initialised = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger under the 'llm-server' namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.

    Example::

        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Server is online.")
    """
    _setup()
    return logging.getLogger(f"llm-server.{name}")