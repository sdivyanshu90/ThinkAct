from __future__ import annotations

from dataclasses import dataclass
import logging

from agent.llm_client import LLMClient
from agent.logging_utils import DebugTracer, log_event
from agent.memory import ConversationMemory
from agent.models import AgentConfig, FinalCommand, MemoryRecord, ParsedCommand, PromptContext, Termination, ToolCallCommand, ToolResult
from agent.parser import CommandParser, ParseFailure
from agent.prompt import build_messages
from agent.safety import RepeatGuard, canonicalize_tool_call, execute_tool_call, sanitize_tool_result
from agent.state import AgentState, ensure_transition
from agent.tool_registry import ToolRegistry


@dataclass(slots=True)
class AgentSession:
    task: str
    state: AgentState = AgentState.INIT
    step_index: int = 0
    raw_model_output: str | None = None
    parsed_command: ParsedCommand | None = None
    pending_tool_result: ToolResult | None = None
    final_answer: str | None = None
    error: str | None = None
    termination_reason: str | None = None


class ReActAgent:
    def __init__(
        self,
        *,
        client: LLMClient,
        registry: ToolRegistry,
        parser: CommandParser,
        config: AgentConfig,
        logger: logging.Logger | None = None,
        tracer: DebugTracer | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._parser = parser
        self._config = config
        self._logger = logger or logging.getLogger("thinkact.agent")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
            self._logger.propagate = False
        self._tracer = tracer or DebugTracer(enabled=False)
        self.memory = ConversationMemory(
            max_messages=config.max_memory_messages,
            max_chars=config.max_memory_chars,
        )
        self._repeat_guard = RepeatGuard(max_consecutive_repeats=config.max_consecutive_repeats)
        self._tool_failures: dict[str, int] = {}

    def run(self, task: str) -> Termination:
        session = AgentSession(task=task.strip())
        if not session.task:
            return Termination(state=AgentState.HALT.value, reason="empty task", steps=0)

        log_event(self._logger, "run_started", task=session.task)
        self._tracer.emit(f"INIT task={session.task}")
        while session.state is not AgentState.HALT:
            if session.state is AgentState.INIT:
                self._handle_init(session)
            elif session.state is AgentState.THINK:
                self._handle_think(session)
            elif session.state is AgentState.ACT:
                self._handle_act(session)
            elif session.state is AgentState.OBSERVE:
                self._handle_observe(session)
            elif session.state is AgentState.REFLECT:
                self._handle_reflect(session)
            elif session.state is AgentState.FINAL:
                self._handle_final(session)
            elif session.state is AgentState.ERROR:
                self._handle_error(session)
            else:
                session.error = f"unhandled state: {session.state.value}"
                session.termination_reason = session.error
                self._transition(session, AgentState.ERROR, session.error)

        termination = Termination(
            state=AgentState.HALT.value,
            reason=session.termination_reason or "halted",
            final_answer=session.final_answer,
            steps=session.step_index,
        )
        log_event(
            self._logger,
            "run_terminated",
            reason=termination.reason,
            steps=termination.steps,
            final_answer=termination.final_answer,
        )
        self._tracer.emit(
            f"HALT steps={termination.steps} reason={termination.reason} final={termination.final_answer or '<none>'}"
        )
        return termination

    def _handle_init(self, session: AgentSession) -> None:
        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.INIT.value,
                user_query=session.task,
            )
        )
        log_event(self._logger, "memory_appended", state=AgentState.INIT.value, turn_index=session.step_index)
        self._transition(session, AgentState.THINK, "session initialized")

    def _handle_think(self, session: AgentSession) -> None:
        if session.step_index >= self._config.max_steps:
            session.termination_reason = "max steps reached"
            self._transition(session, AgentState.ERROR, session.termination_reason)
            return

        session.step_index += 1
        prompt_context = PromptContext(
            user_task=session.task,
            tools=self._registry.specs(),
            memory_window=self.memory.window(),
            last_observation=self.memory.last_observation(),
            step_index=session.step_index,
        )
        messages = build_messages(prompt_context)
        log_event(self._logger, "model_request", step=session.step_index, message_count=len(messages))
        session.raw_model_output = self._client.complete(messages)
        log_event(self._logger, "model_response", step=session.step_index, output=session.raw_model_output)

        try:
            session.parsed_command = self._parser.parse(session.raw_model_output)
        except ParseFailure as exc:
            session.error = str(exc)
            session.termination_reason = "fatal parse error"
            self.memory.append(
                MemoryRecord(
                    turn_index=session.step_index,
                    state=AgentState.THINK.value,
                    model_output=session.raw_model_output,
                    error_info=session.error,
                )
            )
            log_event(self._logger, "parse_error", step=session.step_index, error=session.error)
            self._tracer.emit(f"THINK step={session.step_index} parse_error={session.error}")
            self._transition(session, AgentState.ERROR, session.termination_reason)
            return

        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.THINK.value,
                model_output=session.raw_model_output,
                parsed_command=session.parsed_command,
            )
        )
        log_event(
            self._logger,
            "parsed_command",
            step=session.step_index,
            command_type=session.parsed_command.type,
            thought=getattr(session.parsed_command, "thought", ""),
        )

        if isinstance(session.parsed_command, ToolCallCommand):
            self._tracer.emit(
                f"THINK step={session.step_index} tool={session.parsed_command.tool_name} thought={session.parsed_command.thought}"
            )
            self._transition(session, AgentState.ACT, "tool requested")
            return

        session.final_answer = session.parsed_command.final_answer
        session.termination_reason = "model finalized"
        self._tracer.emit(
            f"THINK step={session.step_index} final thought={session.parsed_command.thought}"
        )
        self._transition(session, AgentState.FINAL, "final command received")

    def _handle_act(self, session: AgentSession) -> None:
        command = session.parsed_command
        if not isinstance(command, ToolCallCommand):
            session.error = "ACT state requires a tool_call command"
            session.termination_reason = session.error
            self._transition(session, AgentState.ERROR, session.error)
            return

        repeat_count = self._repeat_guard.record(command)
        if repeat_count > self._config.max_consecutive_repeats:
            session.error = "repeated identical action detected"
            session.termination_reason = session.error
            log_event(
                self._logger,
                "repeat_detected",
                step=session.step_index,
                tool_name=command.tool_name,
                repeat_count=repeat_count,
            )
            self._tracer.emit(f"ACT step={session.step_index} repeated_action tool={command.tool_name}")
            self._transition(session, AgentState.ERROR, session.error)
            return

        log_event(
            self._logger,
            "tool_call",
            step=session.step_index,
            tool_name=command.tool_name,
            tool_input=command.tool_input,
        )
        self._tracer.emit(f"ACT step={session.step_index} tool={command.tool_name} input={command.tool_input}")
        session.pending_tool_result = execute_tool_call(
            self._registry,
            command,
            timeout_seconds=self._config.tool_timeout_seconds,
        )

        failure_key = canonicalize_tool_call(command)
        if session.pending_tool_result.ok:
            self._tool_failures.pop(failure_key, None)
        else:
            self._tool_failures[failure_key] = self._tool_failures.get(failure_key, 0) + 1
            if self._tool_failures[failure_key] > self._config.max_tool_retries:
                session.error = "tool failure limit reached"
                session.termination_reason = session.error
                log_event(
                    self._logger,
                    "tool_retry_exhausted",
                    step=session.step_index,
                    tool_name=command.tool_name,
                    error=session.pending_tool_result.error,
                )
                self._tracer.emit(
                    f"ACT step={session.step_index} tool_failure_limit tool={command.tool_name} error={session.pending_tool_result.error}"
                )
                self.memory.append(
                    MemoryRecord(
                        turn_index=session.step_index,
                        state=AgentState.ACT.value,
                        model_output=session.raw_model_output,
                        parsed_command=command,
                        tool_result=session.pending_tool_result,
                        error_info=session.error,
                    )
                )
                self._transition(session, AgentState.ERROR, session.error)
                return

        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.ACT.value,
                model_output=session.raw_model_output,
                parsed_command=command,
                tool_result=session.pending_tool_result,
            )
        )
        log_event(
            self._logger,
            "tool_result",
            step=session.step_index,
            ok=session.pending_tool_result.ok,
            error=session.pending_tool_result.error,
        )
        self._transition(session, AgentState.OBSERVE, "tool execution completed")

    def _handle_observe(self, session: AgentSession) -> None:
        session.pending_tool_result = sanitize_tool_result(
            session.pending_tool_result or ToolResult(ok=False, content="", error="missing tool result"),
            max_chars=self._config.max_observation_chars,
        )
        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.OBSERVE.value,
                tool_result=session.pending_tool_result,
            )
        )
        log_event(
            self._logger,
            "observation_sanitized",
            step=session.step_index,
            redacted_lines=session.pending_tool_result.metadata.get("redacted_lines", 0),
        )
        self._tracer.emit(
            f"OBSERVE step={session.step_index} ok={session.pending_tool_result.ok} content={session.pending_tool_result.content}"
        )
        self._transition(session, AgentState.REFLECT, "observation recorded")

    def _handle_reflect(self, session: AgentSession) -> None:
        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.REFLECT.value,
                parsed_command=session.parsed_command,
                tool_result=session.pending_tool_result,
            )
        )
        log_event(self._logger, "reflect", step=session.step_index)
        self._tracer.emit(f"REFLECT step={session.step_index} continue")
        self._transition(session, AgentState.THINK, "continue loop")

    def _handle_final(self, session: AgentSession) -> None:
        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.FINAL.value,
                model_output=session.raw_model_output,
                parsed_command=session.parsed_command,
                final_answer=session.final_answer,
            )
        )
        log_event(self._logger, "final_answer", step=session.step_index, answer=session.final_answer)
        self._tracer.emit(f"FINAL step={session.step_index} answer={session.final_answer}")
        self._transition(session, AgentState.HALT, session.termination_reason or "final answer ready")

    def _handle_error(self, session: AgentSession) -> None:
        self.memory.append(
            MemoryRecord(
                turn_index=session.step_index,
                state=AgentState.ERROR.value,
                model_output=session.raw_model_output,
                parsed_command=session.parsed_command,
                tool_result=session.pending_tool_result,
                error_info=session.error or session.termination_reason,
            )
        )
        log_event(
            self._logger,
            "error",
            step=session.step_index,
            error=session.error,
            reason=session.termination_reason,
        )
        self._tracer.emit(
            f"ERROR step={session.step_index} reason={session.termination_reason} error={session.error}"
        )
        self._transition(session, AgentState.HALT, session.termination_reason or "error")

    def _transition(self, session: AgentSession, new_state: AgentState, reason: str) -> None:
        previous = session.state
        ensure_transition(previous, new_state, reason)
        log_event(self._logger, "state_transition", previous=previous.value, new=new_state.value, reason=reason)
        self._tracer.emit(f"STATE {previous.value} -> {new_state.value}: {reason}")
        session.state = new_state