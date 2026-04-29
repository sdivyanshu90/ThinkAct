from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from agent.llm_client import LLMClient
from agent.models import FinalCommand, ModelMessage, ParsedCommand, ToolCallCommand


@dataclass(slots=True)
class ParseFailure(Exception):
    message: str
    raw_output: str
    attempts: int

    def __str__(self) -> str:
        return f"{self.message} after {self.attempts} attempt(s)"


class CommandParser:
    def __init__(self, client: LLMClient | None = None, max_repair_attempts: int = 1) -> None:
        self._client = client
        self._max_repair_attempts = max_repair_attempts

    def parse(self, raw_output: str) -> ParsedCommand:
        candidate = raw_output
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._parse_once(candidate)
            except ParseFailure as exc:
                if attempts > self._max_repair_attempts or self._client is None:
                    raise ParseFailure(exc.message, raw_output, attempts) from exc
                candidate = self._repair(raw_output, exc.message)

    def _repair(self, raw_output: str, error_message: str) -> str:
        repair_messages = [
            ModelMessage(
                role="system",
                content=(
                    "REPAIR_JSON\n"
                    "Return exactly one valid JSON object matching the required protocol. "
                    "Do not include markdown, prose, or extra keys."
                ),
            ),
            ModelMessage(
                role="user",
                content=(
                    f"ERROR:\n{error_message}\n\n"
                    "MALFORMED_OUTPUT:\n"
                    f"{raw_output}"
                ),
            ),
        ]
        return self._client.complete(repair_messages)

    def _parse_once(self, raw_output: str) -> ParsedCommand:
        if not raw_output or not raw_output.strip():
            raise ParseFailure("model returned empty output", raw_output, 1)

        try:
            payload = json.loads(raw_output)
        except JSONDecodeError as exc:
            raise ParseFailure(f"invalid JSON: {exc.msg}", raw_output, 1) from exc

        if not isinstance(payload, dict):
            raise ParseFailure("top-level JSON value must be an object", raw_output, 1)

        payload_type = payload.get("type")
        if payload_type == "tool_call":
            return self._parse_tool_call(payload, raw_output)
        if payload_type == "final":
            return self._parse_final(payload, raw_output)
        raise ParseFailure("unsupported command type", raw_output, 1)

    def _parse_tool_call(self, payload: dict[str, Any], raw_output: str) -> ToolCallCommand:
        self._ensure_keys(
            payload,
            raw_output,
            allowed={"type", "thought", "tool_name", "tool_input"},
            required={"type", "thought", "tool_name", "tool_input"},
        )
        thought = self._require_string(payload, "thought", raw_output)
        tool_name = self._require_string(payload, "tool_name", raw_output)
        tool_input = payload["tool_input"]
        if not isinstance(tool_input, dict):
            raise ParseFailure("tool_input must be an object", raw_output, 1)
        return ToolCallCommand(
            type="tool_call",
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input,
        )

    def _parse_final(self, payload: dict[str, Any], raw_output: str) -> FinalCommand:
        self._ensure_keys(
            payload,
            raw_output,
            allowed={"type", "thought", "final_answer"},
            required={"type", "thought", "final_answer"},
        )
        thought = self._require_string(payload, "thought", raw_output)
        final_answer = self._require_string(payload, "final_answer", raw_output)
        return FinalCommand(type="final", thought=thought, final_answer=final_answer)

    def _ensure_keys(
        self,
        payload: dict[str, Any],
        raw_output: str,
        *,
        allowed: set[str],
        required: set[str],
    ) -> None:
        missing = required.difference(payload)
        extra = set(payload).difference(allowed)
        if missing:
            raise ParseFailure(f"missing required keys: {sorted(missing)}", raw_output, 1)
        if extra:
            raise ParseFailure(f"unexpected keys: {sorted(extra)}", raw_output, 1)

    def _require_string(self, payload: dict[str, Any], key: str, raw_output: str) -> str:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ParseFailure(f"{key} must be a non-empty string", raw_output, 1)
        return value.strip()