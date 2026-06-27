"""Safety Override Layer: first-pass risk, appropriateness, and abuse screening.

Runs BEFORE phase_manager, context_modifier, and turn_router so safety
constraints are established before any normal routing.

Detects:
  - self-harm risk
  - harm to others / threats / violence
  - coercive control / abuse indicators
  - child safety concerns
  - severe emotional escalation
  - psychiatric / medical red flags
  - situations where couples coaching is inappropriate
"""
import logging

from app.config import RISK_KEYWORDS_HIGH, RISK_KEYWORDS_MEDIUM, COERCIVE_CONTROL_CUES
from app.logutil import ctx_from_state
from app.models import AppState

logger = logging.getLogger(__name__)

RISK_TYPE_KEYWORDS = {
    "self_harm": {"suicide", "kill myself", "end my life", "self-harm", "hurt myself"},
    "abuse": {"abuse", "hit me", "hit her", "hit him", "violent", "scared for my life"},
    "child_safety": {"child abuse", "hurt the kids", "kids are in danger"},
    "stalking": {"stalk", "stalking"},
    "violence": {"violent", "danger", "unsafe", "threaten", "rage", "out of control"},
}

PSYCHIATRIC_RED_FLAGS = {
    "hearing voices", "seeing things", "hallucinating", "psychosis",
    "manic", "bipolar episode", "overdose", "not taking medication",
    "stopped my meds", "voices in my head",
}

SEVERE_ESCALATION_CUES = {
    "i'm going to", "i will hurt", "i'll destroy", "going to kill",
    "smash everything", "break everything", "lose control",
    "can't stop myself", "about to explode", "seeing red",
}


def _classify_risk_type(text: str) -> str:
    for risk_type, kws in RISK_TYPE_KEYWORDS.items():
        if any(k in text for k in kws):
            return risk_type
    return "none"


def _detect_coercive_control(text: str) -> bool:
    return any(cue in text for cue in COERCIVE_CONTROL_CUES)


def _detect_psychiatric_flags(text: str) -> bool:
    return any(flag in text for flag in PSYCHIATRIC_RED_FLAGS)


def _detect_severe_escalation(text: str) -> bool:
    return any(cue in text for cue in SEVERE_ESCALATION_CUES)


def safety_override(state: AppState):
    """First-pass safety screening. Sets risk_level, risk_type, risk_action,
    safety_override_triggered, safety_flags, allowed_actions, must_ask,
    must_refuse, and escalate.
    """
    text = (state.turn.text or "").lower()

    # Defaults
    state.turn.risk_level = "none"
    state.turn.risk_type = "none"
    state.turn.risk_action = "continue"
    state.turn.allowed_actions = ["empathy", "coach", "growth", "general_support"]
    state.turn.must_ask = None
    state.turn.must_refuse = None
    state.turn.escalate = False
    state.turn.safety_override_triggered = False
    state.turn.safety_flags = []

    flags = []

    # --- High risk keywords ---
    for kw in RISK_KEYWORDS_HIGH:
        if kw in text:
            state.turn.risk_level = "high"
            state.turn.risk_action = "escalate_to_human"
            state.turn.risk_type = _classify_risk_type(text) or "abuse"
            state.turn.allowed_actions = ["empathy", "crisis_resources"]
            state.turn.must_refuse = (
                "Do not give relationship advice or exercises; recommend professional help."
            )
            state.turn.escalate = True
            state.turn.safety_override_triggered = True
            flags.append(state.turn.risk_type)
            state.turn.safety_flags = flags
            logger.info(
                "safety_override %s outcome=high risk_type=%s",
                ctx_from_state(state), state.turn.risk_type,
            )
            return {}

    # --- Medium risk keywords ---
    for kw in RISK_KEYWORDS_MEDIUM:
        if kw in text:
            state.turn.risk_level = "medium"
            state.turn.risk_action = "ask_safety_question"
            state.turn.risk_type = _classify_risk_type(text) or "violence"
            state.turn.allowed_actions = ["empathy", "general_support", "safety_question"]
            state.turn.must_ask = "Are you safe right now? Do you have someone you can reach out to?"
            flags.append("medium_risk")
            break

    # --- Coercive control ---
    if _detect_coercive_control(text):
        flags.append("coercive_control")
        state.turn.allowed_actions = ["empathy", "general_support"]
        state.turn.must_refuse = (
            "Do not suggest couples exercises that could increase risk to the "
            "controlled partner; prioritize individual safety."
        )
        if state.turn.risk_level == "none":
            state.turn.risk_level = "medium"
            state.turn.risk_action = "ask_safety_question"
            state.turn.risk_type = "abuse"
            state.turn.must_ask = "Are you safe right now? Do you have someone you can reach out to?"

    # --- Psychiatric red flags ---
    if _detect_psychiatric_flags(text):
        flags.append("psychiatric_red_flag")
        state.turn.must_refuse = (
            "This may require professional psychiatric support beyond what a "
            "couples counseling assistant can provide. Please consider reaching "
            "out to a mental health professional."
        )
        if state.turn.risk_level == "none":
            state.turn.risk_level = "medium"
            state.turn.risk_action = "ask_safety_question"

    # --- Severe escalation ---
    if _detect_severe_escalation(text):
        flags.append("severe_escalation")
        if state.turn.risk_level != "high":
            state.turn.risk_level = "medium"
            state.turn.risk_action = "ask_safety_question"
        state.turn.allowed_actions = ["empathy", "general_support"]
        state.turn.must_ask = state.turn.must_ask or (
            "It sounds like things are very intense right now. Are you and everyone around you safe?"
        )

    # --- Low-level risk type detection (abuse/violence mention without high keywords) ---
    if not flags:
        risk_type = _classify_risk_type(text)
        if risk_type in ("abuse", "violence", "child_safety"):
            flags.append(risk_type)
            state.turn.allowed_actions = ["empathy", "general_support"]
            state.turn.must_refuse = (
                "Do not suggest couples exercises that could increase risk; prioritize safety."
            )
            state.turn.risk_type = risk_type

    if flags:
        state.turn.safety_override_triggered = True
    state.turn.safety_flags = flags

    logger.info(
        "safety_override %s risk_level=%s risk_action=%s flags=%s",
        ctx_from_state(state), state.turn.risk_level,
        state.turn.risk_action, flags,
    )
    return {}


def safety_response(state: AppState):
    """Short-circuit: return crisis resources and do not run coach/growth."""
    logger.info("safety_response %s", ctx_from_state(state))
    state.turn.dialogue_action = "RESPOND_ONLY"
    state.turn.final_response = (
        "I hear that you're going through something very difficult. "
        "Your safety matters. If you're in immediate danger, please contact "
        "emergency services (e.g. 911). For crisis support, you can reach out "
        "to a local crisis line or mental health professional. I'm not able to "
        "provide emergency or clinical care — please consider speaking with a "
        "qualified human professional who can support you."
    )
    return {}
