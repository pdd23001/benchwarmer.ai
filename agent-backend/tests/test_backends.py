"""Tests for LLM backend abstraction."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from benchwarmer.agents.backends import OpenAIBackend, ToolUseBlock, TextBlock


class TestOpenAIBackendConversion(unittest.TestCase):
    """Test message/tool conversion logic without making real API calls."""

    def setUp(self):
        # Mock the openai.OpenAI constructor so no real client is created
        with patch("openai.OpenAI"):
            os.environ["NVIDIA_API_KEY"] = "test-key"
            self.backend = OpenAIBackend(
                base_url="http://test:8000/v1",
                model="test-model",
            )

    def tearDown(self):
        os.environ.pop("NVIDIA_API_KEY", None)

    def test_convert_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello world"}]
        converted = self.backend._convert_messages(messages, "System prompt")

        self.assertEqual(converted[0], {"role": "system", "content": "System prompt"})
        self.assertEqual(converted[1], {"role": "user", "content": "Hello world"})

    def test_convert_assistant_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check..."},
                    {
                        "type": "tool_use", "id": "call_123",
                        "name": "classify_problem", "input": {"description": "max cut"},
                    },
                ],
            }
        ]
        converted = self.backend._convert_messages(messages, "Sys")

        assistant_msg = converted[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(assistant_msg["content"], "Let me check...")
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "call_123")
        self.assertEqual(assistant_msg["tool_calls"][0]["function"]["name"], "classify_problem")

    def test_convert_user_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "Result data"},
                ],
            }
        ]
        converted = self.backend._convert_messages(messages, "")

        tool_msg = converted[1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_123")
        self.assertEqual(tool_msg["content"], "Result data")

    def test_convert_tool_definitions(self):
        anthropic_tool = {
            "name": "classify_problem",
            "description": "Classify the optimization problem",
            "input_schema": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
        converted = self.backend._convert_tool(anthropic_tool)

        self.assertEqual(converted["type"], "function")
        self.assertEqual(converted["function"]["name"], "classify_problem")
        self.assertEqual(converted["function"]["description"], "Classify the optimization problem")
        self.assertEqual(converted["function"]["parameters"]["type"], "object")

    def test_convert_mixed_conversation(self):
        """Full multi-turn conversation with tool use."""
        messages = [
            {"role": "user", "content": "Solve max-cut"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "classify", "input": {"q": "max-cut"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": '{"class": "maximum_cut"}'},
                ],
            },
            {"role": "assistant", "content": "I classified it as maximum_cut."},
        ]
        converted = self.backend._convert_messages(messages, "System")

        # system + user + assistant(tool_calls) + tool + assistant(text)
        self.assertEqual(len(converted), 5)
        self.assertEqual(converted[0]["role"], "system")
        self.assertEqual(converted[1]["role"], "user")
        self.assertEqual(converted[2]["role"], "assistant")
        self.assertIn("tool_calls", converted[2])
        self.assertEqual(converted[3]["role"], "tool")
        self.assertEqual(converted[4]["role"], "assistant")
        self.assertEqual(converted[4]["content"], "I classified it as maximum_cut.")


class TestOpenAIBackendGenerate(unittest.TestCase):
    """Test that generate() correctly wraps OpenAI responses."""

    def setUp(self):
        with patch("openai.OpenAI") as mock_cls:
            self.mock_client = MagicMock()
            mock_cls.return_value = self.mock_client
            os.environ["NVIDIA_API_KEY"] = "test-key"
            self.backend = OpenAIBackend(
                base_url="http://test:8000/v1",
                model="test-model",
            )

    def tearDown(self):
        os.environ.pop("NVIDIA_API_KEY", None)

    def test_generate_text_response(self):
        # Mock a text-only response
        mock_choice = MagicMock()
        mock_choice.message.content = "Here is the config..."
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.backend.generate(
            messages=[{"role": "user", "content": "test"}],
            system="System",
        )

        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(len(result.content), 1)
        self.assertIsInstance(result.content[0], TextBlock)
        self.assertEqual(result.content[0].text, "Here is the config...")

    def test_generate_tool_call_response(self):
        # Mock a tool call response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc"
        mock_tool_call.function.name = "classify_problem"
        mock_tool_call.function.arguments = '{"description": "max cut"}'

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.backend.generate(
            messages=[{"role": "user", "content": "test"}],
            system="System",
            tools=[{
                "name": "classify_problem",
                "description": "Classify",
                "input_schema": {"type": "object", "properties": {}},
            }],
        )

        self.assertEqual(result.stop_reason, "tool_use")
        self.assertEqual(len(result.content), 1)
        self.assertIsInstance(result.content[0], ToolUseBlock)
        self.assertEqual(result.content[0].name, "classify_problem")
        self.assertEqual(result.content[0].input, {"description": "max cut"})


if __name__ == "__main__":
    unittest.main()
