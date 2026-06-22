from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def fetch_response(self, system_prompt: str, user_prompt: str, model: str) -> str:
        ...