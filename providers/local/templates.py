from utils.logger import get_logger

logger = get_logger(__name__)


def _llama3(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _mistral(system_prompt: str, user_prompt: str) -> str:
    return (
        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST]"
    )


def _deepseek(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"system\n{system_prompt}\n"
        f"user\n{user_prompt}\n"
        f"assistant\n"
    )


def _qwen(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


_TEMPLATES = {
    "llama3":   _llama3,
    "mistral":  _mistral,
    "deepseek": _deepseek,
    "qwen":     _qwen,
}


def resolve_prompt(template: str, system_prompt: str, user_prompt: str) -> str:
    fn = _TEMPLATES.get(template)

    if fn is None:
        raise ValueError(f"Unknown prompt template: '{template}'. Available: {list(_TEMPLATES.keys())}")

    logger.info("Resolving prompt with template: %s", template)
    return fn(system_prompt, user_prompt)