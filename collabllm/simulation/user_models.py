from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_USER_SYSTEM_PROMPT = (
    "You are simulating a user in a conversation with an AI assistant. "
    "Based on the conversation history, generate a natural follow-up message as the user. "
    "Stay in character and be consistent with the user's previous messages and goals."
)


class UserModel(ABC):
    """Abstract base class for user simulation models."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or DEFAULT_USER_SYSTEM_PROMPT

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a user response given the conversation history.

        Args:
            messages: Conversation history as list of {"role": str, "content": str}

        Returns:
            Generated user message content
        """
        pass

    def _prepare_messages_for_user_perspective(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Transform messages so the user model sees the conversation from the user's perspective.
        Swaps assistant/user roles so the model generates as if it were the user.
        """
        transformed = [{"role": "system", "content": self.system_prompt}]

        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                transformed.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "assistant":
                transformed.append({"role": "user", "content": msg["content"]})

        return transformed


class OpenAIUserModel(UserModel):
    """User simulation using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        super().__init__(system_prompt)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")

        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        transformed = self._prepare_messages_for_user_perspective(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=transformed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        logger.debug(f"UserModel generated: {content[:100]}...")
        return content
