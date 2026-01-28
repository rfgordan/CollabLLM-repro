# Chat Simulation Module

Simulates multi-turn conversations between a local LLM assistant and an API-based user model for CollabLLM rollouts.

## Components

| File | Purpose |
|------|---------|
| `user_models.py` | Abstract `UserModel` base class + `OpenAIUserModel` implementation |
| `assistant.py` | `LocalAssistant` for loading HF models with optional LoRA/4-bit |
| `simulator.py` | `ChatSimulator` that orchestrates the rollout |

## Usage

```python
from collabllm.simulation import ChatSimulator, LocalAssistant, OpenAIUserModel

# Load local assistant (with optional LoRA/4-bit)
assistant = LocalAssistant(
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    lora_path="./my-lora-adapter",  # optional
    use_4bit=True,                   # optional
)

# Create user simulator (uses OPENAI_API_KEY env var)
user_model = OpenAIUserModel(
    model="gpt-4o-mini",
    system_prompt="You are a curious user asking about machine learning.",
)

# Run simulation
simulator = ChatSimulator(assistant, user_model)
conversation = simulator.rollout(
    conversation_prefix=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is gradient descent?"},
    ],
    max_turns=3,
)
```

## Interactive Mode

For debugging or manual experimentation, use `interactive_rollout`:

```python
# Interactive session - override or auto-generate at each turn
conversation = simulator.interactive_rollout(
    conversation_prefix=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_turns=5,
)
# Press Enter to auto-generate, type to override, '/quit' to exit
```

## Design Notes

- **Extensible user models**: `UserModel` is an abstract base class. To add Anthropic/other providers, subclass it and implement `generate()`.
- **Role swapping**: The user model sees the conversation from the user's perspective (roles are swapped internally so it generates as "assistant" but the output is treated as user messages).
- **Turn definition**: A "turn" = user message + assistant response. `max_turns=3` yields 3 full exchanges.
- **Return format**: Returns the complete messages list including the original prefix.

## Adding a New User Model Provider

```python
from collabllm.simulation.user_models import UserModel

class AnthropicUserModel(UserModel):
    def __init__(self, model: str = "claude-3-haiku-20240307", system_prompt: str = None):
        super().__init__(system_prompt)
        self.model = model
        # Initialize Anthropic client...

    def generate(self, messages: list[dict[str, str]]) -> str:
        transformed = self._prepare_messages_for_user_perspective(messages)
        # Call Anthropic API with transformed messages...
        return response_content
```
