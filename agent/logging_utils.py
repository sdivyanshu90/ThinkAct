from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from typing import Any


def configure_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("thinkact.agent")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    if debug:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.debug(json.dumps({"event": event, **fields}, sort_keys=True, default=str))


@dataclass(slots=True)
class DebugTracer:
    enabled: bool = False

    def emit(self, message: str) -> None:
        if self.enabled:
            print(message, file=sys.stderr)