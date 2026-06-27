"""Shared log context prefix for correlating turns across UI and pipeline."""
from __future__ import annotations

from typing import Any, Mapping, Optional


def ctx(
    turn_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    return f"[turn_id={turn_id or ''} session_id={session_id or ''}]"


def ctx_from_state(state: Any) -> str:
    tid = getattr(getattr(state, "meta", None), "turn_id", None) or ""
    sid = getattr(getattr(state, "therapy", None), "session_id", None) or ""
    return ctx(tid, sid)


def ctx_from_flat(pipeline_input: Optional[Mapping[str, Any]]) -> str:
    if not pipeline_input:
        return ctx()
    return ctx(
        pipeline_input.get("turn_id"),
        pipeline_input.get("session_id"),
    )
