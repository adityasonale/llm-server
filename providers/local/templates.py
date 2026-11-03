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

# ---------------------------------------------------------------------------
# Multi-turn chat prompts, for /v1/chat/completions.
#
# `messages` is a list of {"role": "system" | "user" | "assistant", "content": str}.
# For a single system + user pair each of these produces exactly what the
# single-turn function above produces, so a model behaves the same through
# either endpoint (checked in the __main__ block below).
# ---------------------------------------------------------------------------

def _llama3_chat(messages: list[dict]) -> str:
    turns = "".join(
        f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>" for m in messages
    )
    return f"<|begin_of_text|>{turns}<|start_header_id|>assistant<|end_header_id|>\n"


def _qwen_chat(messages: list[dict]) -> str:
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return f"{turns}<|im_start|>assistant\n"


def _deepseek_chat(messages: list[dict]) -> str:
    turns = "".join(f"{m['role']}\n{m['content']}\n" for m in messages)
    return f"<|begin_of_text|>{turns}assistant\n"


def _mistral_chat(messages: list[dict]) -> str:
    # Mistral has no system role: the system prompt is folded into the first
    # user turn, and each completed assistant turn is closed with </s>.
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    prompt = ""
    first_user = True
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if first_user and system:
                content = f"<<SYS>>\n{system}\n<</SYS>>\n\n{content}"
            first_user = False
            prompt += f"[INST] {content} [/INST]"
        elif m["role"] == "assistant":
            prompt += f" {m['content']}</s>"
    return prompt


_CHAT_TEMPLATES = {
    "llama3":   _llama3_chat,
    "mistral":  _mistral_chat,
    "deepseek": _deepseek_chat,
    "qwen":     _qwen_chat,
}


def resolve_chat_prompt(template: str, messages: list[dict]) -> str:
    fn = _CHAT_TEMPLATES.get(template)

    if fn is None:
        raise ValueError(f"Unknown prompt template: '{template}'. Available: {list(_CHAT_TEMPLATES.keys())}")

    logger.info("Resolving chat prompt with template: %s (%d messages)", template, len(messages))
    return fn(messages)


if __name__ == "__main__":
    # Single-turn parity: both endpoints must render the same prompt.
    pair = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    for name in _TEMPLATES:
        assert resolve_chat_prompt(name, pair) == resolve_prompt(name, "SYS", "USER"), name

    # Multi-turn shape.
    convo = pair + [{"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
    assert _mistral_chat(convo) == "[INST] <<SYS>>\nSYS\n<</SYS>>\n\nUSER [/INST] A1</s>[INST] U2 [/INST]"
    assert _qwen_chat(convo).endswith("<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n")
    assert _mistral_chat([{"role": "user", "content": "hi"}]) == "[INST] hi [/INST]"
    print("templates self-check passed")
