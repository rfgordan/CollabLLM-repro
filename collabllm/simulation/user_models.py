from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "user_simulator.txt"

DEFAULT_TERMINAL_SIGNAL = "[END]"


@dataclass
class UserTurnResult:
    """Structured output from a user simulation turn."""

    response: str
    thought: str = ""
    current_answer: str = ""
    is_terminal: bool = False
    raw_output: str = ""


class UserModel(ABC):
    """Abstract base class for user simulation models."""

    def __init__(
        self,
        task_desc: str,
        single_turn_prompt: str,
        terminal_signal: str = DEFAULT_TERMINAL_SIGNAL,
    ):
        self.task_desc = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.terminal_signal = terminal_signal
        self._prompt_template = PROMPT_TEMPLATE_PATH.read_text()

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> UserTurnResult:
        """
        Generate a user response given the conversation history.

        Args:
            messages: Conversation history as list of {"role": str, "content": str}

        Returns:
            UserTurnResult with structured response fields
        """
        pass

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Fill the prompt template with conversation-specific fields."""
        chat_history = self._format_chat_history(messages)
        return self._prompt_template.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=chat_history,
            terminal_signal=self.terminal_signal,
        )

    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Format message list as text for the prompt template."""
        lines = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            label = "USER" if msg["role"] == "user" else "AI"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines) if lines else "(empty)"

    def _parse_response(self, raw_output: str) -> UserTurnResult:
        """Parse the model's JSON response into a UserTurnResult."""
        try:
            # Strip markdown code fences if present
            text = raw_output.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)

            response = parsed.get("response", "")
            is_terminal = response.strip() == self.terminal_signal

            return UserTurnResult(
                response=response,
                thought=parsed.get("thought", ""),
                current_answer=parsed.get("current_answer", ""),
                is_terminal=is_terminal,
                raw_output=raw_output,
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse user model JSON output: {e}")
            logger.debug(f"Raw output was: {raw_output}")
            return UserTurnResult(
                response=raw_output,
                is_terminal=False,
                raw_output=raw_output,
            )


class OpenAIUserModel(UserModel):
    """User simulation using OpenAI API."""

    def __init__(
        self,
        task_desc: str,
        single_turn_prompt: str,
        terminal_signal: str = DEFAULT_TERMINAL_SIGNAL,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        super().__init__(task_desc, single_turn_prompt, terminal_signal)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")

        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]]) -> UserTurnResult:
        prompt = self._format_prompt(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_output = response.choices[0].message.content
        logger.debug(f"UserModel raw output: {raw_output[:200]}...")

        return self._parse_response(raw_output)
