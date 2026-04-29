from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from agent.models import ToolResult, ToolSpec
from agent.tools import Tool


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unsupported tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def dispatch(self, name: str, input_data: dict[str, object]) -> ToolResult:
        tool = self.get(name)
        return tool.run(dict(input_data))