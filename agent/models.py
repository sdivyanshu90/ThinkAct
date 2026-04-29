from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, frozen=True)
class ModelMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class ToolResult:
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True, frozen=True)
class ToolCallCommand:
    type: Literal["tool_call"]
    thought: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass(slots=True, frozen=True)
class FinalCommand:
    type: Literal["final"]
    thought: str
    final_answer: str


ParsedCommand = ToolCallCommand | FinalCommand


@dataclass(slots=True, frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_parse_retries: int = 1
    max_tool_retries: int = 1
    max_consecutive_repeats: int = 2
    max_memory_messages: int = 12
    max_memory_chars: int = 4_000
    max_observation_chars: int = 1_500
    tool_timeout_seconds: float = 2.0
    debug: bool = False


@dataclass(slots=True, frozen=True)
class MemoryRecord:
    turn_index: int
    state: str
    timestamp: datetime = field(default_factory=utc_now)
    user_query: str | None = None
    model_output: str | None = None
    parsed_command: ParsedCommand | None = None
    tool_result: ToolResult | None = None
    final_answer: str | None = None
    error_info: str | None = None


@dataclass(slots=True, frozen=True)
class Termination:
    state: str
    reason: str
    final_answer: str | None = None
    steps: int = 0


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class SearchDocument:
    title: str
    snippet: str
    url: str


@dataclass(slots=True, frozen=True)
class PromptContext:
    user_task: str
    tools: Sequence[ToolSpec]
    memory_window: Sequence[MemoryRecord]
    last_observation: str | None = None
    step_index: int = 0