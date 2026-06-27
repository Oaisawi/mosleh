"""Context Modifier: classify the session framing for routing policy.

Detects whether the session represents ordinary couple conflict, trust breach,
possible abuse, separation processing, high escalation, one-sided participation,
or individual reflection mode. This framing influences which specialists are safe
to invoke and how the turn router biases its decisions.
"""
import logging

from app.config import CONTEXT_MODIFIER_CUES, COERCIVE_CONTROL_CUES
from app.models import AppState

logger = logging.getLogger(__name__)


def context_modifier(state: AppState):
    """Classify session context modifier from conversation history and current turn.

    Runs after safety_override, before phase_manager.
    Persists on case.context_modifier across turns (sticky until overridden).
    """
    text = (state.turn.text or "").lower()

    history = state.conversation_history or []
    recent_user = " ".join(
        m.get("content", "") for m in history[-8:] if m.get("role") == "user"
    ).lower()
    combined = text + " " + recent_user

    if state.turn.safety_override_triggered:
        if any(f in ("abuse", "violence", "child_safety", "coercive_control")
               for f in state.turn.safety_flags):
            state.case.context_modifier = "possible_abuse"
            logger.info("context_modifier: possible_abuse (safety_flags=%s)", state.turn.safety_flags)
            return {}

    if any(cue in combined for cue in COERCIVE_CONTROL_CUES):
        state.case.context_modifier = "possible_abuse"
        logger.info("context_modifier: possible_abuse (coercive_control_cues)")
        return {}

    for modifier, cues in CONTEXT_MODIFIER_CUES.items():
        if any(cue in combined for cue in cues):
            state.case.context_modifier = modifier
            logger.info("context_modifier: %s", modifier)
            return {}

    if state.case.context_modifier is None:
        state.case.context_modifier = "ordinary_conflict"

    logger.info("context_modifier: %s (no override)", state.case.context_modifier)
    return {}
