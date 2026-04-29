from __future__ import annotations

from dataclasses import dataclass, field

from agent.models import MemoryRecord


@dataclass(slots=True)
class ConversationMemory:
    max_messages: int
    max_chars: int = 4_000
    _records: list[MemoryRecord] = field(default_factory=list)

    def append(self, record: MemoryRecord) -> None:
        self._records.append(record)
        self._trim()

    def window(self) -> list[MemoryRecord]:
        return list(self._records)

    def last_observation(self) -> str | None:
        for record in reversed(self._records):
            if record.tool_result is not None:
                return record.tool_result.content
        return None

    def _trim(self) -> None:
        if not self._records:
            return

        preserved_index = next(
            (index for index, record in enumerate(self._records) if record.user_query),
            None,
        )
        kept: list[MemoryRecord] = []
        current_chars = 0

        for record in reversed(self._records):
            record_size = self._estimate_size(record)
            if kept and (len(kept) >= self.max_messages or current_chars + record_size > self.max_chars):
                continue
            kept.append(record)
            current_chars += record_size

        kept.reverse()

        if preserved_index is not None:
            preserved_record = self._records[preserved_index]
            if preserved_record not in kept:
                if len(kept) >= self.max_messages:
                    kept.pop(0)
                kept.insert(0, preserved_record)

        self._records = kept

    def _estimate_size(self, record: MemoryRecord) -> int:
        parts = [record.state]
        if record.user_query:
            parts.append(record.user_query)
        if record.model_output:
            parts.append(record.model_output)
        if record.parsed_command:
            parts.append(repr(record.parsed_command))
        if record.tool_result:
            parts.append(record.tool_result.content)
            if record.tool_result.error:
                parts.append(record.tool_result.error)
        if record.final_answer:
            parts.append(record.final_answer)
        if record.error_info:
            parts.append(record.error_info)
        return sum(len(part) for part in parts)