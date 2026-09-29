"""
engine_context.py

Execution context for THERM's physics and calculation engine.
Provides contextvars-based debug flag and trace accumulation without importing Streamlit.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

_debug_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "therm_engine_debug_enabled", default=False
)
_debug_traces: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "therm_engine_debug_traces", default=None
)


def set_debug(enabled: bool, traces: Optional[Any] = None) -> None:
    """Set the debug flag and optional traces container (list or dict)."""
    _debug_enabled.set(bool(enabled))
    _debug_traces.set(traces)


def debug_enabled() -> bool:
    """Return True if engine debugging is currently active."""
    return _debug_enabled.get()


def debug_traces() -> Optional[Any]:
    """Return the currently attached traces container, or None."""
    return _debug_traces.get()
