"""Tests for the simulation module."""

import pytest
from typing import List, Dict
from unittest.mock import MagicMock, patch

from collabllm.simulation.user_models import UserModel, DEFAULT_USER_SYSTEM_PROMPT
from collabllm.simulation.simulator import ChatSimulator


class MockUserModel(UserModel):
    """Mock user model for testing."""

    def __init__(self, responses: List[str], system_prompt: str = None):
        super().__init__(system_prompt)
        self.responses = responses
        self.call_count = 0
        self.received_messages = []

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self.received_messages.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockAssistant:
    """Mock assistant for testing."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        self.received_messages = []

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self.received_messages.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestUserModelAbstraction:
    """Tests for the UserModel base class."""

    def test_default_system_prompt(self):
        model = MockUserModel(responses=["test"])
        assert model.system_prompt == DEFAULT_USER_SYSTEM_PROMPT

    def test_custom_system_prompt(self):
        custom_prompt = "You are a test user."
        model = MockUserModel(responses=["test"], system_prompt=custom_prompt)
        assert model.system_prompt == custom_prompt

    def test_perspective_transformation_swaps_roles(self):
        model = MockUserModel(responses=["test"])
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        transformed = model._prepare_messages_for_user_perspective(messages)

        assert transformed[0]["role"] == "system"
        assert transformed[0]["content"] == model.system_prompt
        assert transformed[1]["role"] == "assistant"
        assert transformed[1]["content"] == "Hello"
        assert transformed[2]["role"] == "user"
        assert transformed[2]["content"] == "Hi there!"
        assert transformed[3]["role"] == "assistant"
        assert transformed[3]["content"] == "How are you?"

    def test_perspective_transformation_excludes_original_system(self):
        model = MockUserModel(responses=["test"])
        messages = [
            {"role": "system", "content": "Original system prompt"},
            {"role": "user", "content": "Hello"},
        ]

        transformed = model._prepare_messages_for_user_perspective(messages)

        assert len(transformed) == 2
        assert transformed[0]["content"] == model.system_prompt
        assert "Original system prompt" not in [m["content"] for m in transformed]


class TestChatSimulator:
    """Tests for the ChatSimulator class."""

    def test_rollout_basic(self):
        user = MockUserModel(responses=["User question 1", "User question 2"])
        assistant = MockAssistant(responses=["Assistant response 1", "Assistant response 2"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Initial question"},
        ]

        result = simulator.rollout(prefix, max_turns=2)

        assert len(result) == 7
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Initial question"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert result[4]["role"] == "assistant"
        assert result[5]["role"] == "user"
        assert result[6]["role"] == "assistant"

    def test_rollout_preserves_prefix(self):
        user = MockUserModel(responses=["Follow up"])
        assistant = MockAssistant(responses=["Response"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Original message"},
        ]

        result = simulator.rollout(prefix, max_turns=1)

        assert result[0]["content"] == "System prompt"
        assert result[1]["content"] == "Original message"

    def test_rollout_does_not_modify_original_prefix(self):
        user = MockUserModel(responses=["User msg"])
        assistant = MockAssistant(responses=["Assistant msg"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [{"role": "user", "content": "Hello"}]
        original_len = len(prefix)

        simulator.rollout(prefix, max_turns=1)

        assert len(prefix) == original_len

    def test_should_assistant_go_next_after_user(self):
        user = MockUserModel(responses=["test"])
        assistant = MockAssistant(responses=["test"])
        simulator = ChatSimulator(assistant=assistant, user_model=user)

        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
        ]

        assert simulator._should_assistant_go_next(messages) is True

    def test_should_assistant_go_next_after_assistant(self):
        user = MockUserModel(responses=["test"])
        assistant = MockAssistant(responses=["test"])
        simulator = ChatSimulator(assistant=assistant, user_model=user)

        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        assert simulator._should_assistant_go_next(messages) is False

    def test_rollout_ends_with_assistant(self):
        user = MockUserModel(responses=["Q1", "Q2", "Q3"])
        assistant = MockAssistant(responses=["A1", "A2", "A3"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [{"role": "user", "content": "Start"}]
        result = simulator.rollout(prefix, max_turns=2)

        assert result[-1]["role"] == "assistant"

    def test_rollout_with_start_override(self):
        user = MockUserModel(responses=["User response"])
        assistant = MockAssistant(responses=["Assistant response"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = simulator.rollout(prefix, max_turns=1, start_with_assistant=True)

        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Assistant response"


class TestLocalAssistantWithTinyModel:
    """Tests for LocalAssistant using a minimal HuggingFace model."""

    @pytest.fixture
    def tiny_model_path(self):
        return "hf-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.mark.slow
    def test_local_assistant_initialization(self, tiny_model_path):
        """Test that LocalAssistant can load a tiny model."""
        from collabllm.simulation.assistant import LocalAssistant

        assistant = LocalAssistant(
            model_path=tiny_model_path,
            device_map="cpu",
            use_4bit=False,
        )

        assert assistant.model is not None
        assert assistant.tokenizer is not None

    @pytest.mark.slow
    def test_local_assistant_generate(self, tiny_model_path):
        """Test that LocalAssistant can generate a response."""
        from collabllm.simulation.assistant import LocalAssistant

        assistant = LocalAssistant(
            model_path=tiny_model_path,
            device_map="cpu",
            use_4bit=False,
            max_new_tokens=10,
        )

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        response = assistant.generate(messages)

        assert isinstance(response, str)
        assert len(response) >= 0


class TestOpenAIUserModel:
    """Tests for OpenAIUserModel."""

    def test_openai_user_model_initialization(self):
        """Test OpenAIUserModel can be instantiated with mock client."""
        with patch("openai.OpenAI") as mock_openai:
            from collabllm.simulation.user_models import OpenAIUserModel

            model = OpenAIUserModel(
                model="gpt-4o-mini",
                api_key="test-key",
            )

            assert model.model == "gpt-4o-mini"
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_openai_user_model_generate(self):
        """Test OpenAIUserModel.generate calls the API correctly."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated user message"
            mock_client.chat.completions.create.return_value = mock_response

            from collabllm.simulation.user_models import OpenAIUserModel

            model = OpenAIUserModel(api_key="test-key")

            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]

            result = model.generate(messages)

            assert result == "Generated user message"
            mock_client.chat.completions.create.assert_called_once()

            call_args = mock_client.chat.completions.create.call_args
            sent_messages = call_args.kwargs["messages"]

            assert sent_messages[0]["role"] == "system"
            assert sent_messages[1]["role"] == "assistant"
            assert sent_messages[1]["content"] == "Hello"
            assert sent_messages[2]["role"] == "user"
            assert sent_messages[2]["content"] == "Hi there"
