"""Turn Router: select a primary turn_mode per message, independent of phase.

The turn_mode represents what the current message actually needs:
  safety_check | empathy_containment | clarification | intake_slot_fill |
  psychoeducation | communication_coaching | trust_repair | closeness_building |
  maintenance_review | progress_reflection

Phase biases which modes are preferred, but the router decides based on the
actual message content, emotional state, and safety context.
"""
import logging

from app.config import (
    ACTION_CUES,
    GREETINGS,
    PLAN_CUES,
    UNDERSTANDING_CUES,
    PATTERN_CUES,
    FOUR_HORSEMEN_CUES,
    PHASE_POLICIES,
    RISK_KEYWORDS_HIGH,
)
from app.models import AppState

logger = logging.getLogger(__name__)

DISTRESS_CUES = {
    "frustrated", "angry", "upset", "overwhelmed", "can't take", "tired of",
    "hopeless", "exhausted", "falling apart", "crying", "scared", "miserable",
    "fed up", "giving up", "breaking down", "had enough", "so done",
    "can't anymore", "lost", "helpless", "furious", "devastated",
    "divorce", "break up", "breaking up", "ending", "leave me", "leaving",
    "separation", "separate", "can't do this anymore", "what's the point",
    "giving up on us", "it's over", "done with this", "i'm done",
}

HELP_NOW_CUES = {
    "help me", "what should i do", "what do i do", "i don't know what to do",
    "what is doable", "give me something", "suggest", "any tips",
    "what can i try", "how do i", "what should", "step by step",
    "step-by-step", "concrete", "practical",
}

DISSATISFACTION_CUES = {
    "only solution", "only option", "that's it", "that it",
    "anything else", "something else", "something different",
    "already tried that", "tried that", "didn't work",
    "not helpful", "not working", "doesn't help",
    "is that all", "nothing new", "same thing",
}

RESISTANCE_CUES = {
    "too much", "a lot of work", "lot of work", "too complicated",
    "simpler", "easier", "less work", "overwhelmed by",
    "can't do all that", "that's a lot", "sounds like a lot",
    "not realistic", "too many steps", "too structured",
}

SHAME_DIGNITY_CUES = {
    "feel small", "feeling small", "makes me feel small",
    "feel stupid", "feeling stupid", "i'm stupid", "i am stupid",
    "not worth", "worth the effort", "not enough", "asking for too much",
    "begging for attention", "begging for basic attention", "begging",
    "teach someone to care", "teach him to care", "teach her to care",
    "shouldn't have to teach", "should not have to teach",
    "shouldn't have to ask", "should not have to ask",
    "shouldn't have to beg", "should not have to beg",
    "i shouldn't have to", "i should not have to",
    "it feels unfair", "feels unfair", "unfair",
    "hate that feeling", "hate feeling",
}

REQUIRED_SLOTS = {"situation_summary", "who_involved", "timeframe", "what_tried", "desired_outcome"}


def _slots_completeness(slots_filled: dict) -> float:
    filled = sum(1 for k in REQUIRED_SLOTS if slots_filled.get(k))
    return filled / len(REQUIRED_SLOTS)


def _is_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return True
    words = t.split()
    return len(words) <= 3 and any(g in t for g in GREETINGS)


def _has_shame_dignity_cue(text: str) -> bool:
    """Detect worth, shame, or dignity pain that needs attunement before tools."""
    return any(cue in text for cue in SHAME_DIGNITY_CUES)


def turn_router(state: AppState):
    """Classify the current message into a turn_mode.

    Runs after phase_manager, before specialist_orchestrator.
    Sets turn.turn_mode, turn.turn_mode_reason, turn.turn_type (legacy compat),
    turn.emotional_intensity, turn.response_style, turn.needs_rag, and
    turn.detected_horsemen.
    """
    text = (state.turn.text or "").lower()
    phase = state.therapy.current_phase
    policy = PHASE_POLICIES.get(phase, PHASE_POLICIES[1])
    slots = state.case.slots_filled or {}
    completeness = _slots_completeness(slots)
    questions_asked = state.case.questions_asked or 0
    context_mod = state.case.context_modifier or "ordinary_conflict"
    fallback = state.therapy.temporary_fallback

    blocked = policy.get("blocked", set())

    # Detect signals
    is_distressed = any(c in text for c in DISTRESS_CUES)
    wants_help_now = any(c in text for c in HELP_NOW_CUES)
    is_dissatisfied = any(c in text for c in DISSATISFACTION_CUES)
    is_resisting = any(c in text for c in RESISTANCE_CUES)
    has_understanding = any(c in text for c in UNDERSTANDING_CUES)
    has_pattern = any(c in text for c in PATTERN_CUES)
    has_coach = any(c in text for c in ACTION_CUES)
    has_growth = any(c in text for c in PLAN_CUES)
    has_shame_dignity = _has_shame_dignity_cue(text)
    is_greeting = _is_greeting(text)

    emotion_cues = [
        "i feel", "i'm feeling", "i am feeling", "worried", "overwhelmed",
        "sad", "angry", "anxious", "scared", "hurt", "lonely", "alone",
        "frustrated", "exhausted", "tired", "drained", "broken",
    ]
    has_emotion = any(c in text for c in emotion_cues)

    detected_horsemen = [
        h for h, cues in FOUR_HORSEMEN_CUES.items()
        if any(c in text for c in cues)
    ]
    state.turn.detected_horsemen = detected_horsemen
    if detected_horsemen:
        state.case.conflict_pattern_assessment = (
            f"Detected conflict markers: {', '.join(detected_horsemen)}"
        )

    # --- Priority 1: Safety override already triggered upstream ---
    if state.turn.safety_override_triggered:
        _set(state, "safety_check", "safety_override_active",
             "high_risk_escalation", 1.0, "empathy_only", False)
        return {}

    # --- Priority 2: High risk keywords (redundant check for robustness) ---
    if any(kw in text for kw in RISK_KEYWORDS_HIGH):
        _set(state, "safety_check", "risk_keywords_detected",
             "high_risk_escalation", 1.0, "empathy_only", False)
        return {}

    # --- Priority 3: Temporary fallback or abuse context -> containment ---
    if fallback:
        _set(state, "empathy_containment",
             f"temporary_fallback_active",
             "venting_emotion", 0.6, "empathy_only", False)
        return {}
    if context_mod == "possible_abuse":
        _set(state, "empathy_containment",
             f"context={context_mod}",
             "venting_emotion", 0.8, "empathy_only", False)
        return {}

    # --- Priority 4: Shame/dignity pain -> attunement before theory or tools ---
    if has_shame_dignity and not (wants_help_now or has_understanding or has_pattern):
        _set(state, "empathy_containment", "shame_dignity_cue",
             "venting_emotion", 0.85, "empathy_only", False)
        return {}

    # --- Priority 4: High distress without help-now request ---
    if is_distressed and not wants_help_now:
        _set(state, "empathy_containment", "distress_detected",
             "venting_emotion", 0.8, "empathy_only", False)
        return {}

    # --- Priority 5: Understanding / why questions ---
    if has_understanding or has_pattern:
        mode = "psychoeducation"
        _set(state, mode, "understanding_or_pattern_cues",
             "understanding", 0.4 if has_emotion else 0.2, "understanding",
             True)
        return {}

    # --- Priority 6: Greeting / low-context -> intake ---
    if is_greeting and questions_asked == 0:
        _set(state, "intake_slot_fill", "greeting_first_turn",
             "intake_needed", 0.1, "empathy_only", False)
        return {}

    if completeness < 0.4 and questions_asked < 6 and not (wants_help_now or is_dissatisfied):
        _set(state, "intake_slot_fill", f"low_completeness={completeness:.1f}",
             "intake_needed", 0.3 if has_emotion else 0.1, "empathy_only", False)
        return {}

    # --- Priority 7: User resisting workload -> low-burden empathy path ---
    if is_resisting:
        _set(state, "empathy_containment", "resistance_to_workload",
             "understanding", 0.3 if has_emotion else 0.2,
             "empathy_light_advice", False)
        return {}

    # --- Priority 8: Dissatisfaction -> coaching only if phase allows ---
    if is_dissatisfied:
        if "communication_coaching" not in blocked:
            _set(state, "communication_coaching", "dissatisfaction_detected",
                 "advice_coaching", 0.4 if has_emotion else 0.2, "full_advice",
                 completeness >= 0.2)
        else:
            _set(state, "psychoeducation",
                 f"dissatisfaction_but_coaching_blocked_phase_{phase}",
                 "understanding", 0.3, "understanding", completeness >= 0.2)
        return {}

    # --- Priority 9: Direct help request ---
    if wants_help_now or has_coach:
        if "communication_coaching" not in blocked:
            _set(state, "communication_coaching", "help_request_or_action_cues",
                 "advice_coaching", 0.5 if has_emotion else 0.2,
                 "empathy_light_advice" if has_emotion else "full_advice",
                 completeness >= 0.3)
        else:
            _set(state, "clarification",
                 f"coaching_blocked_in_phase_{phase}_offering_clarification",
                 "advice_coaching", 0.5 if has_emotion else 0.2,
                 "empathy_light_advice", False)
        return {}

    # --- Priority 10: Growth plan ---
    if has_growth:
        if phase >= 4:
            _set(state, "maintenance_review", "growth_cues_later_phase",
                 "growth_plan", 0.3, "full_advice", completeness >= 0.3)
        elif "communication_coaching" not in blocked:
            _set(state, "communication_coaching", "growth_cues_early_phase",
                 "advice_coaching", 0.3, "empathy_light_advice", True)
        else:
            _set(state, "psychoeducation",
                 f"growth_but_coaching_blocked_phase_{phase}",
                 "understanding", 0.3, "understanding", True)
        return {}

    # --- Priority 11: Phase-biased defaults ---
    if phase <= 1:
        _set(state, "intake_slot_fill", "phase_1_default",
             "intake_needed", 0.3 if has_emotion else 0.1, "empathy_only", False)
    elif phase == 2:
        _set(state, "psychoeducation", "phase_2_default",
             "understanding", 0.3, "understanding", True)
    elif phase == 3:
        mode = "communication_coaching" if not has_emotion else "empathy_containment"
        _set(state, mode, "phase_3_default",
             "mixed", 0.4, "empathy_light_advice", True)
    elif phase == 4:
        _set(state, "trust_repair", "phase_4_default",
             "mixed", 0.3, "empathy_light_advice", True)
    else:
        _set(state, "maintenance_review", "phase_5_default",
             "mixed", 0.3, "full_advice", True)

    return {}


def _set(state: AppState, turn_mode: str, reason: str,
         turn_type: str, intensity: float, response_style: str,
         needs_rag: bool):
    """Helper to set all turn-routing fields at once."""
    state.turn.turn_mode = turn_mode
    state.turn.turn_mode_reason = reason
    state.turn.turn_type = turn_type
    state.turn.emotional_intensity = intensity
    state.turn.response_style = response_style
    state.turn.needs_rag = needs_rag
    # Preserve user_intent for downstream compat
    if turn_mode in ("communication_coaching", "maintenance_review"):
        state.turn.user_intent = "coach"
    elif turn_mode == "psychoeducation":
        state.turn.user_intent = "understand"
    elif turn_mode in ("empathy_containment", "safety_check"):
        state.turn.user_intent = "vent"
    else:
        state.turn.user_intent = state.turn.user_intent or "mixed"

    logger.info(
        "turn_router: mode=%s reason=%s turn_type=%s intensity=%.1f style=%s",
        turn_mode, reason, turn_type, intensity, response_style,
    )
