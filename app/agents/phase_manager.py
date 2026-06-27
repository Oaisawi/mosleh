"""Phase Manager: determines current therapy phase, evaluates transitions, and
provides phase-specific guidance to downstream agents.

The 5-phase couples therapy program:
  1. Assessment & Building Therapeutic Relationship
  2. Understanding Self & Partner
  3. Communication Skills & Conflict Management
  4. Building Trust & Emotional Closeness
  5. Stabilization & Prevention

Adaptive progression:
  - Milestone completion is weighted, not binary.
  - Soft readiness signals provide supporting evidence.
  - Turn count is a weak signal, never decisive.
  - Hard-cap turns trigger review_needed, not forced advance.
  - Supports stay / advance / temporary_fallback / regress / review_needed.
"""
import logging
from typing import Dict, List, Optional, Tuple

from app.config import (
    MODEL_NAME,
    THERAPY_PHASES,
    THERAPY_APPROACHES,
    SOFT_READINESS_SIGNALS,
)
from app.llm.providers import ask_model
from app.models import AppState

logger = logging.getLogger(__name__)


def _phase_config(phase: int) -> dict:
    return THERAPY_PHASES.get(phase, THERAPY_PHASES[1])


def _milestone_progress(milestones: Dict[str, bool], phase: int) -> float:
    cfg = _phase_config(phase)
    expected = cfg.get("milestones", [])
    if not expected:
        return 1.0
    achieved = sum(1 for m in expected if milestones.get(m, False))
    return achieved / len(expected)


def _evaluate_milestones_with_llm(state: AppState) -> Dict[str, bool]:
    """Use LLM to evaluate which milestones have been achieved in the current phase."""
    phase = state.therapy.current_phase
    cfg = _phase_config(phase)
    expected_milestones = cfg.get("milestones", [])
    if not expected_milestones:
        return state.therapy.milestones or {}

    current = state.therapy.milestones or {}
    unchecked = [m for m in expected_milestones if not current.get(m, False)]
    if not unchecked:
        return current

    history = state.conversation_history or []
    recent = history[-10:]
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent
    )

    slots_info = ""
    if state.case.slots_filled:
        slots_info = "\n".join(f"- {k}: {v}" for k, v in state.case.slots_filled.items() if v)

    milestones_str = "\n".join(f"- {m}" for m in unchecked)

    system_prompt = (
        "You are evaluating therapy progress for a couples counseling session.\n\n"
        f"Current therapy phase: {phase} - {cfg['name_en']}\n"
        f"Phase objectives: {', '.join(cfg['objectives'])}\n\n"
        f"Collected information:\n{slots_info}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Milestones to evaluate (has this been addressed in the conversation?):\n{milestones_str}\n\n"
        "For each milestone, output one line: MILESTONE_NAME: YES or NO\n"
        "Say YES if the conversation has addressed this topic with reasonable depth.\n"
        "A direct answer from the user counts — they do not need to write a paragraph.\n"
        "For example, if the user described their problem and who is involved, "
        "'problem_assessed' and 'safety_established' can be YES."
    )

    raw = ask_model(system_prompt, user_prompt=state.turn.text or "", model=MODEL_NAME)

    updated = dict(current)
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip().upper()
            if key in expected_milestones and val == "YES":
                updated[key] = True

    return updated


def _detect_soft_signals(state: AppState) -> List[str]:
    """Detect soft readiness signals from recent conversation history and turn text."""
    signals = []
    text = (state.turn.text or "").lower()
    history = state.conversation_history or []
    recent_text = " ".join(
        m.get("content", "") for m in history[-6:] if m.get("role") == "user"
    ).lower()
    combined = text + " " + recent_text

    if any(w in combined for w in ("i realize", "i see now", "i understand", "my part in", "i contributed")):
        signals.append("reflects_on_own_role")
    if any(w in combined for w in ("she feels", "he feels", "their perspective", "from their side", "partner feels")):
        signals.append("considers_partner_perspective")
    if any(w in combined for w in ("i feel", "i'm feeling", "makes me feel", "i've been feeling")):
        signals.append("describes_feelings_clearly")
    if any(w in combined for w in ("i'll try", "willing to", "open to", "i can try", "let me try")):
        signals.append("open_to_practical_tools")
    if any(w in combined for w in ("that helped", "it worked", "useful", "it was good", "made a difference")):
        signals.append("reports_prior_advice_useful")
    if any(w in combined for w in ("calmer now", "feeling better", "more settled", "less angry", "cooled down")):
        signals.append("emotional_regulation")
    if any(w in combined for w in ("maybe i should", "i could have", "next time i'll")):
        signals.append("willing_to_try_new_approach")

    intensity = state.turn.emotional_intensity or 0.0
    if intensity < 0.3 and text and len(text) > 20:
        if "emotional_regulation" not in signals:
            signals.append("emotional_regulation")

    return signals


def _compute_readiness_evidence(state: AppState) -> Tuple[float, str]:
    """Compute weighted readiness score from four evidence channels.

    Weights (sum = 1.0):
      milestones  0.35  — LLM-evaluated milestone completion
      soft        0.20  — phrase-based readiness signals from conversation
      engagement  0.20  — turn count relative to min_turns (caps at 1.0)
      structural  0.25  — slot completeness + intake_completed (system state)

    Returns (score 0..1, human-readable reason).
    """
    phase = state.therapy.current_phase
    milestones = state.therapy.milestones or {}
    milestone_pct = _milestone_progress(milestones, phase)

    soft_signals = _detect_soft_signals(state)
    state.case.soft_signals_detected = soft_signals
    soft_pct = len(soft_signals) / max(len(SOFT_READINESS_SIGNALS), 1)

    cfg = _phase_config(phase)
    min_turns = cfg.get("min_turns", 3)
    turns = state.therapy.turns_in_phase
    turn_pct = min(turns / max(min_turns, 1), 1.0)

    slots = state.case.slots_filled or {}
    required_slots = {"situation_summary", "who_involved", "timeframe",
                      "what_tried", "desired_outcome"}
    slot_fill_pct = sum(1 for k in required_slots if slots.get(k)) / len(required_slots)
    intake_done = 1.0 if state.case.intake_completed else 0.0
    structural_pct = slot_fill_pct * 0.6 + intake_done * 0.4

    score = (
        milestone_pct * 0.35
        + soft_pct * 0.20
        + turn_pct * 0.20
        + structural_pct * 0.25
    )
    score = round(min(score, 1.0), 3)

    completed = [m for m, v in milestones.items() if v]
    state.case.milestones_completed = completed

    reason_parts = [f"milestones={milestone_pct:.0%}({len(completed)})"]
    if soft_signals:
        reason_parts.append(f"soft_signals={soft_pct:.0%}({','.join(soft_signals[:3])})")
    reason_parts.append(f"turns={turns}/{min_turns}")
    reason_parts.append(f"structural={structural_pct:.0%}")

    return score, " | ".join(reason_parts)


def _decide_transition(state: AppState, readiness: float, reason: str) -> Tuple[str, str]:
    """Decide phase transition: stay / advance / temporary_fallback / regress / review_needed.

    Rules:
      - Phase 5 never advances.
      - Advance requires readiness >= 0.50 AND turns >= min_turns.
      - Hard cap (min_turns * 2) triggers review_needed, NOT forced advance.
      - High emotional intensity in later phases triggers temporary_fallback.
      - Regression requires strong evidence (safety concern or sustained distress).
    """
    phase = state.therapy.current_phase
    cfg = _phase_config(phase)
    min_turns = cfg.get("min_turns", 3)
    turns = state.therapy.turns_in_phase
    intensity = state.turn.emotional_intensity or 0.0
    safety_override = state.turn.safety_override_triggered

    if phase >= 5:
        return "stay", f"phase_5_terminal | {reason}"

    if safety_override and phase > 1:
        return "temporary_fallback", f"safety_override_active | {reason}"

    if intensity >= 0.8 and phase > 2:
        return "temporary_fallback", f"high_distress_intensity={intensity:.1f} | {reason}"

    if turns < min_turns:
        return "stay", f"below_min_turns({turns}<{min_turns}) | {reason}"

    if readiness >= 0.50:
        return "advance", f"readiness={readiness:.2f}>=0.50 | {reason}"

    hard_cap = min_turns * 2
    if turns >= hard_cap:
        return "review_needed", f"hard_cap_reached({turns}>={hard_cap}) readiness={readiness:.2f} | {reason}"

    return "stay", f"readiness={readiness:.2f}<0.50 | {reason}"


def _apply_transition(state: AppState, decision: str):
    """Apply the transition decision to therapy state."""
    if decision == "advance":
        old_phase = state.therapy.current_phase
        new_phase = min(old_phase + 1, 5)

        state.therapy.phase_history.append({
            "phase": old_phase,
            "turns": state.therapy.turns_in_phase,
            "milestones": dict(state.therapy.milestones or {}),
            "transition": "advance",
        })

        state.therapy.current_phase = new_phase
        state.therapy.turns_in_phase = 0
        state.therapy.milestones = {}
        state.therapy.phase_notes = None
        state.therapy.temporary_fallback = False

        logger.info(
            "phase_manager: PHASE ADVANCE %d -> %d (total_turns=%d)",
            old_phase, new_phase, state.therapy.total_turns,
        )

    elif decision == "temporary_fallback":
        state.therapy.temporary_fallback = True
        logger.info(
            "phase_manager: TEMPORARY FALLBACK phase=%d (total_turns=%d)",
            state.therapy.current_phase, state.therapy.total_turns,
        )

    elif decision == "regress":
        old_phase = state.therapy.current_phase
        new_phase = max(old_phase - 1, 1)
        state.therapy.phase_history.append({
            "phase": old_phase,
            "turns": state.therapy.turns_in_phase,
            "milestones": dict(state.therapy.milestones or {}),
            "transition": "regress",
        })
        state.therapy.current_phase = new_phase
        state.therapy.turns_in_phase = 0
        state.therapy.milestones = {}
        state.therapy.phase_notes = None
        state.therapy.temporary_fallback = False

        logger.info(
            "phase_manager: PHASE REGRESS %d -> %d (total_turns=%d)",
            old_phase, new_phase, state.therapy.total_turns,
        )

    elif decision == "review_needed":
        state.therapy.temporary_fallback = False
        logger.info(
            "phase_manager: REVIEW NEEDED phase=%d turns=%d (total_turns=%d)",
            state.therapy.current_phase, state.therapy.turns_in_phase,
            state.therapy.total_turns,
        )

    else:  # stay
        intensity = state.turn.emotional_intensity or 0.0
        if state.therapy.temporary_fallback and intensity < 0.5:
            state.therapy.temporary_fallback = False
            logger.info("phase_manager: clearing temporary_fallback (intensity=%.1f)", intensity)


def _get_phase_guidance(state: AppState) -> str:
    """Generate therapist-style phase guidance for downstream agents."""
    phase = state.therapy.current_phase
    cfg = _phase_config(phase)

    milestones = state.therapy.milestones or {}
    achieved = [m for m in cfg.get("milestones", []) if milestones.get(m)]
    pending = [m for m in cfg.get("milestones", []) if not milestones.get(m)]

    approach = THERAPY_APPROACHES.get(
        state.therapy.therapy_approach or "integrative",
        THERAPY_APPROACHES["integrative"],
    )

    guidance_parts = [
        f"THERAPY PHASE {phase}/5: {cfg['name_en']} ({cfg['name_ar']})",
        f"Phase focus: {cfg['description']}",
        f"Objectives: {', '.join(cfg['objectives'])}",
        f"Tools/techniques to use: {', '.join(cfg['tools'])}",
        f"Therapy approach: {approach['name_en']} — {approach['description']}",
        "Therapist stance: collaborative, emotionally attuned, formulation-led, non-judgmental.",
    ]

    if state.therapy.temporary_fallback:
        guidance_parts.append(
            "NOTE: This turn is in TEMPORARY FALLBACK mode — prioritize containment, "
            "empathy, and safety over phase-specific interventions."
        )

    if achieved:
        guidance_parts.append(f"Milestones achieved: {', '.join(achieved)}")
    if pending:
        guidance_parts.append(f"Milestones still needed: {', '.join(pending)}")
        guidance_parts.append(f"Priority: work toward '{pending[0]}' in this response")

    soft = state.case.soft_signals_detected or []
    if soft:
        guidance_parts.append(f"Soft readiness signals observed: {', '.join(soft)}")

    return "\n".join(guidance_parts)


def phase_manager(state: AppState):
    """Main phase manager node. Evaluates milestones, decides transitions,
    and sets phase guidance for downstream agents.

    Called after safety_override and context_modifier, before turn_router.
    """
    state.therapy.total_turns += 1
    state.therapy.turns_in_phase += 1

    if state.therapy.turns_in_phase > 1:
        state.therapy.milestones = _evaluate_milestones_with_llm(state)

    readiness, reason = _compute_readiness_evidence(state)
    decision, full_reason = _decide_transition(state, readiness, reason)

    state.therapy.phase_transition_decision = decision
    state.therapy.phase_transition_reason = full_reason
    state.therapy.phase_confidence = readiness

    _apply_transition(state, decision)

    state.therapy.phase_notes = _get_phase_guidance(state)

    logger.info(
        "phase_manager: phase=%d decision=%s turns_in_phase=%d total=%d readiness=%.3f milestones=%s fallback=%s",
        state.therapy.current_phase,
        decision,
        state.therapy.turns_in_phase,
        state.therapy.total_turns,
        readiness,
        state.therapy.milestones,
        state.therapy.temporary_fallback,
    )
    return {}


def get_phase_context_for_prompt(state: AppState) -> str:
    """Return a compact phase-context block for inclusion in agent prompts."""
    phase = state.therapy.current_phase
    cfg = _phase_config(phase)
    approach = THERAPY_APPROACHES.get(
        state.therapy.therapy_approach or "integrative",
        THERAPY_APPROACHES["integrative"],
    )

    progress = _milestone_progress(state.therapy.milestones or {}, phase)
    milestones = state.therapy.milestones or {}
    pending = [m for m in cfg.get("milestones", []) if not milestones.get(m)]

    lines = [
        f"[Phase {phase}/5: {cfg['name_en']}]",
        f"Focus: {cfg['description']}",
        f"Approach: {approach['name_en']}",
        f"Progress: {progress:.0%}",
    ]
    if state.therapy.temporary_fallback:
        lines.append("Mode: TEMPORARY FALLBACK — prioritize containment and empathy")
    if pending:
        lines.append(f"Next milestone: {pending[0]}")
    decision = state.therapy.phase_transition_decision
    if decision and decision != "stay":
        lines.append(f"Transition: {decision}")
    return "\n".join(lines)
