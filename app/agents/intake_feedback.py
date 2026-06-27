"""Post-intake therapist feedback/evaluation node.

Builds a concise therapist-style formulation before the support path begins.
"""
import logging

from app.logutil import ctx_from_state
from app.models import AppState

logger = logging.getLogger(__name__)


def _top_focus_areas(slots: dict) -> list[str]:
    mapping = {
        "conflict_triggers": "conflict triggers",
        "how_arguments_end": "conflict endings",
        "repair_attempts": "repair attempts",
        "partner_perspective": "partner perspective",
        "desired_outcome": "shared outcome",
    }
    focus = []
    for key, label in mapping.items():
        if slots.get(key):
            focus.append(label)
    return focus[:3]


def intake_feedback_evaluation(state: AppState):
    """Create structured therapist feedback after intake readiness is met."""
    slots = state.case.slots_filled or {}
    logger.info(
        "intake_feedback_evaluation %s slots_keys_n=%s",
        ctx_from_state(state),
        len(slots),
    )
    if not slots:
        logger.info(
            "intake_feedback_evaluation %s skip reason=empty_slots",
            ctx_from_state(state),
        )
        return {}

    summary_parts = []
    if slots.get("relationship_length"):
        summary_parts.append(f"Relationship length: {slots.get('relationship_length')}")
    if slots.get("major_transitions"):
        summary_parts.append(f"Transitions: {slots.get('major_transitions')}")
    if slots.get("situation_summary"):
        summary_parts.append(f"Presenting concern: {slots.get('situation_summary')}")
    if slots.get("conflict_triggers"):
        summary_parts.append(f"Trigger pattern: {slots.get('conflict_triggers')}")
    state.case.formulation_summary = "; ".join(summary_parts)[:500] or state.case.formulation_summary

    strengths = slots.get("relationship_strengths")
    if strengths:
        state.case.strengths_summary = strengths

    focus_areas = _top_focus_areas(slots)
    if focus_areas:
        state.case.focus_areas = focus_areas
        state.case.plan = (
            f"Therapist focus: {', '.join(focus_areas)} | "
            f"phase={state.therapy.current_phase} | mode={state.case.therapy_mode}"
        )

    # If a conflict pattern has enough detail, store a compact assessment string.
    if slots.get("conflict_triggers") or slots.get("how_arguments_end"):
        state.case.conflict_pattern_assessment = (
            f"Trigger: {slots.get('conflict_triggers', 'n/a')} | "
            f"Argument end: {slots.get('how_arguments_end', 'n/a')} | "
            f"Repair: {slots.get('repair_attempts', 'n/a')}"
        )[:500]
    logger.info(
        "intake_feedback_evaluation %s formulation_set=%s strengths_set=%s focus_n=%s",
        ctx_from_state(state),
        bool(state.case.formulation_summary),
        bool(state.case.strengths_summary),
        len(state.case.focus_areas or []),
    )
    return {}
