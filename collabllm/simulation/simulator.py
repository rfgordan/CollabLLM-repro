from typing import List, Dict, Optional
import logging
from copy import deepcopy

from collabllm.simulation.user_models import UserModel
from collabllm.simulation.assistant import LocalAssistant

logger = logging.getLogger(__name__)


class ChatSimulator:
    """
    Simulates multi-turn conversations between a local assistant and a user model.

    The simulator alternates between assistant and user turns, starting from
    a conversation prefix and continuing until max_turns is reached.
    """

    def __init__(
        self,
        assistant: LocalAssistant,
        user_model: UserModel,
    ):
        """
        Initialize the chat simulator.

        Args:
            assistant: Local LLM to act as the assistant
            user_model: API-based model to simulate the user
        """
        self.assistant = assistant
        self.user_model = user_model

    def rollout(
        self,
        conversation_prefix: List[Dict[str, str]],
        max_turns: int,
        start_with_assistant: Optional[bool] = None,
    ) -> List[Dict[str, str]]:
        """
        Continue a conversation from a prefix until max_turns is reached.

        A "turn" is one exchange (user message + assistant response).

        Args:
            conversation_prefix: Initial messages to start from.
                Format: [{"role": "system"|"user"|"assistant", "content": str}, ...]
            max_turns: Maximum number of turns to simulate.
                Each turn consists of one user message and one assistant response.
            start_with_assistant: Override which role goes first.
                If None, automatically determined from the last message in prefix.

        Returns:
            Complete conversation as list of messages (prefix + generated)
        """
        messages = deepcopy(conversation_prefix)

        if start_with_assistant is None:
            start_with_assistant = self._should_assistant_go_next(messages)

        turns_completed = 0
        assistant_turn = start_with_assistant

        while turns_completed < max_turns:
            if assistant_turn:
                response = self.assistant.generate(messages)
                messages.append({"role": "assistant", "content": response})
                logger.info(f"Turn {turns_completed + 1}: Assistant responded")
            else:
                response = self.user_model.generate(messages)
                messages.append({"role": "user", "content": response})
                logger.info(f"Turn {turns_completed + 1}: User responded")

            if not assistant_turn:
                turns_completed += 1

            assistant_turn = not assistant_turn

        # Ensure conversation ends with assistant response
        if assistant_turn:
            response = self.assistant.generate(messages)
            messages.append({"role": "assistant", "content": response})
            logger.info("Final assistant response added")

        return messages

    def _should_assistant_go_next(self, messages: List[Dict[str, str]]) -> bool:
        """Determine if assistant should respond next based on last non-system message."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                return True
            elif msg["role"] == "assistant":
                return False
        return True

    def interactive_rollout(
        self,
        conversation_prefix: List[Dict[str, str]],
        max_turns: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Interactive chat session with option to override or auto-generate responses.

        At each turn, you can:
        - Press Enter to auto-generate the response
        - Type a custom response to use instead
        - Type '/quit' or '/q' to end the session early

        Args:
            conversation_prefix: Initial messages to start from.
            max_turns: Maximum turns before auto-stopping (default 10).

        Returns:
            Complete conversation as list of messages
        """
        messages = deepcopy(conversation_prefix)
        assistant_turn = self._should_assistant_go_next(messages)
        turns_completed = 0

        print("\n" + "=" * 60)
        print("Interactive Chat Session")
        print("- Press Enter to auto-generate")
        print("- Type your own response to override")
        print("- Type '/quit' or '/q' to exit")
        print("=" * 60)

        # Display prefix
        for msg in messages:
            self._print_message(msg)

        while turns_completed < max_turns:
            role = "assistant" if assistant_turn else "user"
            print(f"\n[{role.upper()}] ", end="", flush=True)

            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break

            if user_input.lower() in ("/quit", "/q"):
                print("Session ended.")
                break

            if user_input:
                response = user_input
            else:
                print("  (generating...)", end="\r")
                if assistant_turn:
                    response = self.assistant.generate(messages)
                else:
                    response = self.user_model.generate(messages)
                print(f"  {response}")

            messages.append({"role": role, "content": response})

            if not assistant_turn:
                turns_completed += 1

            assistant_turn = not assistant_turn

        print("\n" + "=" * 60)
        print(f"Session complete. {len(messages)} messages total.")
        print("=" * 60)

        return messages

    def _print_message(self, msg: Dict[str, str]) -> None:
        """Print a formatted message."""
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"\n[{role}] {content}")
