from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from agent.models import ToolCallCommand, ToolResult
from agent.tool_registry import ToolRegistry


_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"tool\s*:\s*", re.IGNORECASE),
    re.compile(r"type\s*:\s*final", re.IGNORECASE),
    re.compile(r"call\s+the\s+tool", re.IGNORECASE),
)


def canonicalize_tool_call(command: ToolCallCommand) -> str:
    return json.dumps(
        {
            "tool_name": command.tool_name,
            "tool_input": command.tool_input,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(slots=True)
class RepeatGuard:
    max_consecutive_repeats: int
    _last_signature: str | None = None
    _repeat_count: int = 0

    def record(self, command: ToolCallCommand) -> int:
        signature = canonicalize_tool_call(command)
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1
        return self._repeat_count


def sanitize_tool_result(result: ToolResult, max_chars: int) -> ToolResult:
    content, redacted_lines = _sanitize_text(result.content, max_chars)
    error, _ = _sanitize_text(result.error or "", max_chars)
    metadata = dict(result.metadata)
    metadata["sanitized"] = True
    metadata["redacted_lines"] = redacted_lines
    return ToolResult(ok=result.ok, content=content, metadata=metadata, error=error or None)


def execute_tool_call(
    registry: ToolRegistry,
    command: ToolCallCommand,
    timeout_seconds: float,
) -> ToolResult:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(registry.dispatch, command.tool_name, command.tool_input)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        future.cancel()
        return ToolResult(
            ok=False,
            content="",
            metadata={"timeout": True},
            error=f"tool timeout after {timeout_seconds:.2f}s",
        )
    except KeyError as exc:
        return ToolResult(ok=False, content="", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive boundary
        return ToolResult(ok=False, content="", error=f"tool execution failed: {exc}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _sanitize_text(text: str, max_chars: int) -> tuple[str, int]:
    if not text:
        return "", 0

    redacted_lines = 0
    sanitized_lines: list[str] = []
    for line in text.splitlines() or [text]:
        if any(pattern.search(line) for pattern in _PROMPT_INJECTION_PATTERNS):
            sanitized_lines.append("[redacted prompt-injection-like content]")
            redacted_lines += 1
        else:
            sanitized_lines.append(line)

    sanitized = "\n".join(sanitized_lines).strip()
    if not sanitized:
        sanitized = "EMPTY_OBSERVATION"
    if len(sanitized) > max_chars:
        sanitized = sanitized[: max_chars - len("...[truncated]")] + "...[truncated]"
    return sanitized, redacted_lines