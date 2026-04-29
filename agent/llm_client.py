from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Sequence

from agent.models import ModelMessage


class LLMClient(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: Sequence[ModelMessage]) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class QueuedLLMClient(LLMClient):
    responses: list[str]
    name: str = "queued-mock"
    requests: list[list[ModelMessage]] = field(default_factory=list)

    def complete(self, messages: Sequence[ModelMessage]) -> str:
        self.requests.append(list(messages))
        if not self.responses:
            raise RuntimeError("QueuedLLMClient has no remaining responses")
        return self.responses.pop(0)


@dataclass(slots=True)
class RuleBasedMockLLMClient(LLMClient):
    name: str = "rule-based-mock"
    requests: list[list[ModelMessage]] = field(default_factory=list)

    def complete(self, messages: Sequence[ModelMessage]) -> str:
        self.requests.append(list(messages))
        joined = "\n".join(message.content for message in messages)
        if "REPAIR_JSON" in joined:
            return self._repair_response(joined)
        return self._plan_response(joined)

    def _repair_response(self, prompt_text: str) -> str:
        match = re.search(r"MALFORMED_OUTPUT:\n(?P<output>.*)", prompt_text, re.DOTALL)
        malformed_output = match.group("output").strip() if match else ""
        first_brace = malformed_output.find("{")
        last_brace = malformed_output.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = malformed_output[first_brace : last_brace + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                return candidate
        return json.dumps(
            {
                "type": "final",
                "thought": "repair fallback",
                "final_answer": "Unable to repair the previous model output.",
            }
        )

    def _plan_response(self, prompt_text: str) -> str:
        task = self._extract_section(prompt_text, "USER_TASK")
        observation = self._extract_section(prompt_text, "LAST_OBSERVATION")
        if observation == "NONE":
            observation = ""

        expression = self._extract_expression(task)
        if expression and not observation:
            return json.dumps(
                {
                    "type": "tool_call",
                    "thought": "use calculator for arithmetic",
                    "tool_name": "calculator",
                    "tool_input": {"expression": expression},
                }
            )
        if expression and observation:
            return json.dumps(
                {
                    "type": "final",
                    "thought": "calculator result is available",
                    "final_answer": f"The result is {observation}.",
                }
            )

        if observation:
            if observation.startswith("NO_RESULTS"):
                answer = "I could not find external information with the configured search provider."
            else:
                first_line = observation.splitlines()[0]
                answer = f"Based on the search results: {first_line}"
            return json.dumps(
                {
                    "type": "final",
                    "thought": "search observation is sufficient",
                    "final_answer": answer,
                }
            )

        return json.dumps(
            {
                "type": "tool_call",
                "thought": "use search when external information may be needed",
                "tool_name": "search",
                "tool_input": {"query": task},
            }
        )

    def _extract_section(self, prompt_text: str, name: str) -> str:
        match = re.search(rf"{name}:\n(?P<value>.*?)(?:\n[A-Z_]+:\n|\Z)", prompt_text, re.DOTALL)
        return match.group("value").strip() if match else ""

    def _extract_expression(self, task: str) -> str | None:
        task = task.strip()
        exact = re.search(r"(?:what is|calculate|compute)\s+([-+*/().%\d\s]+)\??$", task, re.IGNORECASE)
        if exact:
            return exact.group(1).strip()

        generic = re.fullmatch(r"[-+*/().%\d\s]+", task)
        if generic:
            return task.strip()

        return None


@dataclass(slots=True, frozen=True)
class OpenAICompatibleProviderConfig:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float = 30.0
    app_name: str = "ThinkAct"
    app_url: str = "https://example.local/thinkact"
    require_free_model_suffix: bool = False
    free_model_suffix: str = ":free"


@dataclass(slots=True)
class OpenAICompatibleLLMClient(LLMClient):
    config: OpenAICompatibleProviderConfig
    name: str = "openai-compatible"
    requests: list[list[ModelMessage]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.config.api_key.strip():
            raise ValueError("provider api key must be non-empty")
        if not self.config.model.strip():
            raise ValueError("provider model must be non-empty")
        if self.config.require_free_model_suffix and not self.config.model.endswith(self.config.free_model_suffix):
            raise ValueError(
                f"model '{self.config.model}' is not allowed; expected suffix '{self.config.free_model_suffix}'"
            )

    def complete(self, messages: Sequence[ModelMessage]) -> str:
        self.requests.append(list(messages))
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            self._completion_url(),
            data=body,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider request failed with HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"provider request failed: {exc.reason}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"provider returned unexpected response shape: {payload}") from exc
        return self._extract_text(content)

    def _completion_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.config.base_url:
            headers["HTTP-Referer"] = self.config.app_url
            headers["X-OpenRouter-Title"] = self.config.app_name
        return headers

    def _extract_text(self, content: object) -> str:
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            joined = "".join(parts).strip()
            if joined:
                return joined

        raise RuntimeError(f"provider returned empty content: {content!r}")