"""Specialist Orchestrator: decide which specialists to invoke per turn.

Replaces the old _support_entry_node() hard phase-locks with policy-driven
selection based on:
  - phase policy (preferred / allowed / blocked)
  - turn_mode
  - safety flags and context modifier
  - coaching readiness (emotional state + context completeness)
"""
import logging

from app.config import PHASE_POLICIES, CATEGORY_MODALITIES
from app.models import AppState

logger = logging.getLogger(__name__)

_COACHING_TURN_MODES = {
    "communication_coaching", "trust_repair", "closeness_building",
    "maintenance_review",
}

_UNDERSTANDING_TURN_MODES = {
    "psychoeducation", "clarification", "progress_reflection",
}

_RAG_TURN_MODES = {
    "psychoeducation",
    "communication_coaching",
    "trust_repair",
    "closeness_building",
    "maintenance_review",
    "progress_reflection",
}

_RAG_SKIP_MODES = {
    "safety_check",
    "empathy_containment",
    "intake_slot_fill",
    "clarification",
}


def _compute_coaching_readiness(state: AppState) -> tuple[bool, str]:
    """Gate coaching by emotional readiness and context completeness.

    Returns (eligible, reason).
    """
    intensity = state.turn.emotional_intensity or 0.0
    turn_mode = state.turn.turn_mode or ""
    context_mod = state.case.context_modifier or "ordinary_conflict"
    safety_flags = state.turn.safety_flags or []
    slots = state.case.slots_filled or {}
    required = {"situation_summary", "who_involved", "timeframe", "what_tried", "desired_outcome"}
    filled = sum(1 for k in required if slots.get(k))
    slot_readiness = filled / len(required)

    # Block coaching entirely for abuse/coercive contexts
    if context_mod == "possible_abuse" or any(
        f in ("abuse", "coercive_control", "violence") for f in safety_flags
    ):
        return False, f"unsafe_context={context_mod}"

    if intensity >= 0.65:
        return False, f"too_dysregulated(intensity={intensity:.1f})"

    if slot_readiness < 0.2 and turn_mode not in ("communication_coaching",):
        return False, f"insufficient_context(slots={slot_readiness:.1f})"

    profile = (state.profile.profile_notes or "").strip()
    profile_ok = len(profile) > 40 and profile.lower() not in {
        "no profile yet.", "not captured yet.", "none yet."
    }
    if slot_readiness < 0.4 and not profile_ok:
        return False, f"low_readiness(slots={slot_readiness:.1f},no_profile)"

    return True, f"eligible(intensity={intensity:.1f},slots={slot_readiness:.1f})"


def _compute_theory_eligibility(state: AppState) -> tuple[bool, str]:
    """Gate pattern/psychoeducation so insight does not interrupt attunement."""
    intensity = state.turn.emotional_intensity or 0.0
    turn_mode = state.turn.turn_mode or ""
    style = (state.turn.response_style or "").lower()
    context_mod = state.case.context_modifier or "ordinary_conflict"
    safety_flags = state.turn.safety_flags or []

    if state.turn.safety_override_triggered:
        return False, "safety_override"
    if context_mod == "possible_abuse" or any(
        f in ("abuse", "coercive_control", "violence") for f in safety_flags
    ):
        return False, f"unsafe_context={context_mod}"
    if turn_mode == "empathy_containment" or style == "empathy_only":
        return False, f"attunement_first(mode={turn_mode})"
    if turn_mode == "psychoeducation":
        if intensity >= 0.85:
            return False, f"too_dysregulated_for_theory(intensity={intensity:.1f})"
        return True, f"explicit_understanding_request(intensity={intensity:.1f})"
    if intensity >= 0.55:
        return False, f"too_activated_for_theory(intensity={intensity:.1f})"
    return True, f"eligible_for_theory(intensity={intensity:.1f})"


def _compute_readiness_score(state: AppState) -> tuple[float, str]:
    """Compute an overall readiness score from slots + profile + history."""
    slots = state.case.slots_filled or {}
    required = {"situation_summary", "who_involved", "timeframe", "what_tried", "desired_outcome"}
    filled = sum(1 for k in required if slots.get(k))
    readiness = filled / len(required)

    profile = (state.profile.profile_notes or "").strip()
    if len(profile) > 80 and profile.lower() not in {"no profile yet.", "not captured yet."}:
        readiness = min(1.0, readiness + 0.3)

    readiness = round(readiness, 2)
    reason = f"slots={filled}/{len(required)} profile_len={len(profile)}"
    return readiness, reason


def _should_run_rag(state: AppState, readiness: float) -> bool:
    """Use Qdrant only for grounded support, not intake/safety/attunement."""
    turn_mode = state.turn.turn_mode or ""
    context_mod = state.case.context_modifier or "ordinary_conflict"
    safety_flags = state.turn.safety_flags or []

    if readiness < 0.3:
        return False
    if state.turn.safety_override_triggered or state.therapy.temporary_fallback:
        return False
    if turn_mode in _RAG_SKIP_MODES:
        return False
    if context_mod == "possible_abuse" or any(
        flag in ("abuse", "coercive_control", "violence") for flag in safety_flags
    ):
        return False
    return turn_mode in _RAG_TURN_MODES


def specialist_orchestrator(state: AppState):
    """Decide run_* flags based on turn_mode, phase policy, safety, and readiness.

    Runs after turn_router, before support chain (rag -> understanding -> specialists).
    """
    phase = state.therapy.current_phase
    policy = PHASE_POLICIES.get(phase, PHASE_POLICIES[1])
    turn_mode = state.turn.turn_mode or "empathy_containment"
    context_mod = state.case.context_modifier or "ordinary_conflict"
    safety_override = state.turn.safety_override_triggered
    fallback = state.therapy.temporary_fallback

    # Compute readiness
    readiness, readiness_reason = _compute_readiness_score(state)
    state.case.readiness_score = readiness
    state.case.readiness_reason = readiness_reason

    # Compute coaching eligibility
    coaching_ok, coaching_reason = _compute_coaching_readiness(state)
    state.case.coaching_eligible = coaching_ok
    state.case.coaching_eligibility_reason = coaching_reason

    # Compute theory eligibility separately from coaching readiness.
    theory_ok, theory_reason = _compute_theory_eligibility(state)

    # --- Safety override: empathy only ---
    if safety_override or turn_mode == "safety_check":
        _set_all(state, emotion=True)
        state.case.plan = f"safety_override | {readiness_reason}"
        logger.info("specialist_orchestrator: safety_override -> emotion_only")
        return {}

    # --- Temporary fallback: mostly empathy, maybe psychoeducation ---
    if fallback:
        _set_all(state, emotion=True, psychoeducation=(turn_mode in _UNDERSTANDING_TURN_MODES))
        state.case.plan = f"fallback | {readiness_reason}"
        logger.info("specialist_orchestrator: fallback -> emotion + maybe psychoeducation")
        return {}

    # --- Abuse context: empathy + maybe psychoeducation, no coaching ---
    if context_mod == "possible_abuse":
        _set_all(state, emotion=True, psychoeducation=theory_ok)
        state.case.plan = f"possible_abuse | {readiness_reason}"
        logger.info("specialist_orchestrator: possible_abuse -> emotion + psychoeducation")
        return {}

    # --- Normal orchestration by turn_mode ---
    run_emotion = False
    run_coach = False
    run_growth = False
    run_psychoeducation = False
    run_pattern = False
    run_rag = False

    if turn_mode in ("empathy_containment", "safety_check"):
        run_emotion = True

    elif turn_mode == "intake_slot_fill":
        run_emotion = True

    elif turn_mode == "clarification":
        run_emotion = True

    elif turn_mode == "psychoeducation":
        run_emotion = True
        run_psychoeducation = theory_ok
        run_pattern = theory_ok

    elif turn_mode == "communication_coaching":
        run_emotion = (state.turn.emotional_intensity or 0.0) >= 0.3
        run_coach = coaching_ok
        run_psychoeducation = theory_ok and "psychoeducation" not in policy["blocked"]

    elif turn_mode == "trust_repair":
        run_emotion = True
        run_coach = coaching_ok
        run_psychoeducation = theory_ok

    elif turn_mode == "closeness_building":
        run_emotion = True
        run_growth = coaching_ok and phase >= 4
        run_coach = coaching_ok

    elif turn_mode in ("maintenance_review", "progress_reflection"):
        run_emotion = (state.turn.emotional_intensity or 0.0) >= 0.4
        run_coach = coaching_ok
        run_growth = coaching_ok and phase >= 4

    else:
        run_emotion = True

    # Category overlay (additive, never overrides safety/coaching blocks)
    category = (state.case.problem_category or "").strip()
    cat_mods = CATEGORY_MODALITIES.get(category)
    if cat_mods and coaching_ok and theory_ok:
        if cat_mods.get("psychoeducation") and not run_psychoeducation:
            if "psychoeducation" not in policy.get("blocked", set()):
                run_psychoeducation = True
        if cat_mods.get("pattern") and not run_pattern:
            run_pattern = True

    # Phase policy enforcement: block modes that are explicitly blocked
    blocked = policy.get("blocked", set())
    if "communication_coaching" in blocked:
        run_coach = False
    if "trust_repair" in blocked and turn_mode == "trust_repair":
        run_coach = False
        run_growth = False
    if "maintenance_review" in blocked and turn_mode == "maintenance_review":
        run_growth = False

    # Allowed-but-limited: if turn_mode is in allowed_limited, cap specialist count
    allowed_limited = policy.get("allowed_limited", set())
    if turn_mode in allowed_limited:
        if run_coach and run_growth:
            run_growth = False

    run_rag = _should_run_rag(state, readiness)

    _set_all(state,
             emotion=run_emotion,
             coach=run_coach,
             growth=run_growth,
             psychoeducation=run_psychoeducation,
             pattern=run_pattern,
             rag=run_rag)

    # Build plan string
    plan_parts = []
    for name, flag in [("emotion", run_emotion), ("coach", run_coach),
                       ("growth", run_growth), ("psychoeducation", run_psychoeducation),
                       ("pattern", run_pattern), ("rag", run_rag)]:
        if flag:
            plan_parts.append(name)
    state.case.plan = (
        f"Run: {', '.join(plan_parts) or 'none'} | mode={turn_mode} | "
        f"readiness={readiness:.2f} | coaching={coaching_ok} | theory={theory_ok}"
    )

    logger.info(
        "specialist_orchestrator: mode=%s agents=[%s] readiness=%.2f coaching=%s theory=%s context=%s phase=%d reason=%s",
        turn_mode, ", ".join(plan_parts), readiness, coaching_ok, theory_ok, context_mod, phase, theory_reason,
    )
    return {}


def _set_all(state: AppState, *,
             emotion: bool = False, coach: bool = False, growth: bool = False,
             psychoeducation: bool = False, pattern: bool = False,
             rag: bool = False):
    """Set all run_* flags and sync legacy accessors."""
    state.turn.run_emotion = emotion
    state.turn.run_coach = coach
    state.turn.run_growth = growth
    state.turn.run_psychoeducation = psychoeducation
    state.turn.run_pattern = pattern
    state.turn.run_rag = rag
    state.turn.needs_rag = rag
    state.need_emotion = emotion
    state.need_coach = coach
    state.need_growth = growth
    state.need_psychoeducation = psychoeducation
    state.need_pattern = pattern
