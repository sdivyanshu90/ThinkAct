from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.llm_client import OpenAICompatibleLLMClient, OpenAICompatibleProviderConfig
from agent.models import ModelMessage


class _FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class OpenAICompatibleLLMClientTests(unittest.TestCase):
    def test_posts_chat_completion_and_returns_text(self) -> None:
        captured_request = None

        def fake_urlopen(request, timeout=0):
            nonlocal captured_request
            captured_request = request
            self.assertEqual(timeout, 5.0)
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"type":"final","thought":"done","final_answer":"ok"}'
                            }
                        }
                    ]
                }
            )

        client = OpenAICompatibleLLMClient(
            name="openrouter",
            config=OpenAICompatibleProviderConfig(
                api_key="test-key",
                model="meta-llama/llama-3.3-8b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                timeout_seconds=5.0,
                require_free_model_suffix=True,
            ),
        )

        with patch("agent.llm_client.urllib_request.urlopen", side_effect=fake_urlopen):
            result = client.complete([ModelMessage(role="user", content="What is 2+2?")])

        self.assertEqual(result, '{"type":"final","thought":"done","final_answer":"ok"}')
        self.assertIsNotNone(captured_request)
        self.assertEqual(captured_request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(captured_request.get_method(), "POST")
        self.assertEqual(captured_request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(captured_request.headers["X-openrouter-title"], "ThinkAct")

    def test_rejects_non_free_openrouter_model(self) -> None:
        with self.assertRaises(ValueError):
            OpenAICompatibleLLMClient(
                name="openrouter",
                config=OpenAICompatibleProviderConfig(
                    api_key="test-key",
                    model="openai/gpt-5.2",
                    base_url="https://openrouter.ai/api/v1",
                    require_free_model_suffix=True,
                ),
            )