from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentState(str, Enum):
    INIT = "INIT"
    THINK = "THINK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    REFLECT = "REFLECT"
    FINAL = "FINAL"
    ERROR = "ERROR"
    HALT = "HALT"


_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.INIT: {AgentState.THINK, AgentState.ERROR, AgentState.HALT},
    AgentState.THINK: {AgentState.ACT, AgentState.FINAL, AgentState.ERROR, AgentState.HALT},
    AgentState.ACT: {AgentState.OBSERVE, AgentState.ERROR, AgentState.HALT},
    AgentState.OBSERVE: {AgentState.REFLECT, AgentState.ERROR, AgentState.HALT},
    AgentState.REFLECT: {AgentState.THINK, AgentState.FINAL, AgentState.ERROR, AgentState.HALT},
    AgentState.FINAL: {AgentState.HALT},
    AgentState.ERROR: {AgentState.HALT},
    AgentState.HALT: set(),
}


@dataclass(slots=True, frozen=True)
class StateTransition:
    previous: AgentState
    new: AgentState
    reason: str


def ensure_transition(previous: AgentState, new: AgentState, reason: str) -> StateTransition:
    if new not in _ALLOWED_TRANSITIONS[previous]:
        raise ValueError(f"Invalid state transition: {previous} -> {new}")
    return StateTransition(previous=previous, new=new, reason=reason)