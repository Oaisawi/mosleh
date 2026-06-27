"""Coach agent: solution-oriented practical advice for couples counseling.

This agent analyzes the user's specific situation (what's been tried, constraints,
partner personality) and produces a TAILORED SOLUTION — not a generic exercise.
It adapts depth based on response_style and avoids repeating prior advice.
"""
import re
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState, CoachOutput
from app.utils import profile_is_collected


def _parse_coach_output(raw: str, fallback_confidence: float = 0.75) -> CoachOutput:
    """Parse LLM response into CoachOutput."""
    exercise_title = duration = warning_notes = None
    steps = []
    confidence = fallback_confidence
    in_steps = False
    for line in raw.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lower = line_stripped.lower()
        if lower.startswith("exercise_title") or lower.startswith("title") or lower.startswith("solution_title"):
            exercise_title = line_stripped.split(":", 1)[-1].strip()
        elif lower.startswith("step ") or lower.startswith("steps"):
            in_steps = True
            val = line_stripped.split(":", 1)[-1].strip()
            if val:
                steps.append(val)
        elif lower.startswith("duration"):
            duration = line_stripped.split(":", 1)[-1].strip()
        elif lower.startswith("warning"):
            warning_notes = line_stripped.split(":", 1)[-1].strip()
        elif lower.startswith("confidence"):
            try:
                confidence = float(re.search(r"[\d.]+", line_stripped).group(0))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, AttributeError):
                pass
        elif in_steps and (line_stripped[0].isdigit() or line_stripped.startswith("-")):
            steps.append(line_stripped.lstrip("- ").strip())
    if not exercise_title and raw.strip():
        exercise_title = raw.strip()[:80]
    if not steps and raw.strip():
        steps = [raw.strip()]
    return CoachOutput(
        exercise_title=exercise_title,
        steps=steps,
        duration=duration,
        warning_notes=warning_notes,
        confidence=confidence,
    )


def _build_situation_context(slots_filled: dict) -> str:
    """Build a concise situation summary from filled slots for the coach prompt."""
    if not slots_filled:
        return ""
    parts = []
    mapping = {
        "situation_summary": "Situation",
        "who_involved": "People involved",
        "timeframe": "How long",
        "what_tried": "Already tried",
        "desired_outcome": "What they want",
    }
    for key, label in mapping.items():
        val = slots_filled.get(key)
        if val and str(val).strip():
            parts.append(f"- {label}: {val}")
    # Include any extra slots (partner personality, constraints, etc.)
    for key, val in slots_filled.items():
        if key not in mapping and val and str(val).strip():
            parts.append(f"- {key.replace('_', ' ').title()}: {val}")
    return "\n".join(parts)


def _extract_prior_advice(conversation_history: list) -> str:
    """Extract assistant messages that contain advice to avoid repetition."""
    if not conversation_history:
        return ""
    prior = []
    for msg in conversation_history[-10:]:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if len(content) > 50:  # skip very short messages (greetings, questions)
                prior.append(content[:200])
    if not prior:
        return ""
    return "\n---\n".join(prior[-3:])  # last 3 substantive assistant messages


SHAME_DIGNITY_CUES = {
    "feel small", "feeling small", "not worth", "worth the effort",
    "asking for too much", "begging", "teach someone to care",
    "shouldn't have to", "should not have to", "feel stupid", "feeling stupid",
}


def _has_shame_dignity_cue(text: str) -> bool:
    return any(cue in (text or "").lower() for cue in SHAME_DIGNITY_CUES)


def coach_agent_compute(
    text: str,
    problem_category: str,
    profile_notes: str,
    conversation_history: list,
    readiness_score: float = 0.0,
    response_style: str = "full_advice",
    slots_filled: dict = None,
    retrieved_info: str = "",
) -> dict:
    """
    Pure compute: LLM call + parse. Returns dict with coach_output, coach_response.
    Used by parallel_specialists and by coach_agent.

    response_style controls output depth:
      - "empathy_light_advice": ONE brief, situation-specific suggestion (1-2 sentences)
      - "full_advice" (default): full tailored solution with analysis and approach
    """
    if not profile_is_collected(profile_notes) or readiness_score < 0.4:
        return {}
    if (
        not text
        or not problem_category
        or (problem_category or "").lower() == "not enough"
    ):
        return {}

    situation_context = _build_situation_context(slots_filled or {})
    prior_advice = _extract_prior_advice(conversation_history)
    shame_dignity_context = _has_shame_dignity_cue(text)
    knowledge_context = ""
    if retrieved_info.strip():
        knowledge_context = (
            "\n## Retrieved knowledge context\n"
            "Use this only if relevant to ground the suggestion. Do not quote it "
            "verbatim, and do not mention retrieval or sources in the answer.\n"
            f"{retrieved_info.strip()}\n"
        )

    # Build the no-repeat rule
    no_repeat_rule = ""
    if prior_advice:
        no_repeat_rule = (
            "\n## CRITICAL: Do NOT repeat prior advice\n"
            "The following advice was ALREADY given in this conversation. "
            "You MUST propose something DIFFERENT — build on it, go deeper, or address a new angle.\n"
            f"Prior advice given:\n{prior_advice}\n"
            "---\n"
        )

    no_script_rule = ""
    if response_style == "empathy_light_advice" or shame_dignity_context:
        no_script_rule = (
            "\n## Attunement timing\n"
            "The user may be feeling shame, low worth, or dignity pain. Do NOT provide "
            "a verbatim dialogue script such as 'tell him...' or 'say to her...'. "
            "Prefer one gentle principle or an optional wording idea, and keep it secondary "
            "to validation.\n"
        )

    # Choose prompt depth based on response_style
    if response_style == "empathy_light_advice":
        system_prompt = (
            "## Role\n"
            f"You are a relationship counselor specializing in {problem_category}.\n"
            "\n## User's Situation\n"
            f"{situation_context or 'See conversation history.'}\n"
            f"Profile: {profile_notes}\n"
            f"{knowledge_context}"
            "\n## Task\n"
            "Give ONE brief, specific suggestion in 1-2 sentences that is TAILORED to this person's "
            "exact situation — what they've tried, their partner's personality, their constraints. "
            "No generic techniques. No structured exercises. No weekly plans. No bullet points.\n"
            + no_repeat_rule +
            no_script_rule +
            "\n## Output format\n"
            "Exercise_title: <short label>\n"
            "Step 1: <your 1-2 sentence tailored suggestion>\n"
            "Confidence: <0.0 to 1.0>\n"
        )
    else:
        system_prompt = (
            "## Role\n"
            f"You are a relationship counselor specializing in {problem_category}. "
            "You provide TAILORED SOLUTIONS, not generic exercises.\n"
            "\n## User's Situation\n"
            f"{situation_context or 'See conversation history.'}\n"
            f"Profile: {profile_notes}\n"
            f"{knowledge_context}"
            "\n## Task\n"
            "Propose ONE specific, tailored approach for this couple.\n"
            "A good approach:\n"
            "- Acknowledges WHY this problem exists for THIS couple\n"
            "- Proposes something concrete that fits their reality and constraints\n"
            "- Adapts to what HASN'T worked (don't repeat failed strategies)\n"
            "Keep it brief: one core idea, explained clearly in 2-3 sentences. "
            "Not a multi-step protocol.\n"
            "\n## Do not\n"
            "- Do NOT give generic advice ('try communicating more')\n"
            "- Do NOT suggest things similar to what already failed\n"
            "- Do NOT give clinical/emergency advice\n"
            "- Do NOT output long paragraphs, weekly timelines, or numbered protocols\n"
            + no_repeat_rule +
            no_script_rule +
            "\n## Output format\n"
            "Solution_title: <short label for the approach>\n"
            "Step 1: <the core suggestion — WHY it fits and WHAT to try, in 2-3 sentences>\n"
            "Step 2: <one follow-up thought, optional>\n"
            "Warning_notes: <any caveat, or blank>\n"
            "Confidence: <0.0 to 1.0>\n"
        )

    raw = ask_model(
        system_prompt,
        user_prompt=text,
        model=MODEL_NAME,
        history=conversation_history or [],
    )
    out = _parse_coach_output(raw)
    response_text = out.exercise_title or ""
    if out.steps:
        response_text += "\n" + "\n".join(f"- {s}" for s in out.steps)
    return {
        "coach_output": out,
        "coach_response": response_text,
    }


def coach_agent(state: AppState):
    """Provide solution-oriented advice tailored to the user's specific situation."""
    readiness = getattr(state.case, "readiness_score", 0.0) or 0.0
    if not profile_is_collected(state.profile_notes) or readiness < 0.4:
        return {}
    if (
        not state.text
        or not state.problem_category
        or not state.need_coach
        or (state.problem_category or "").lower() == "not enough"
    ):
        return {}
    result = coach_agent_compute(
        state.text,
        state.problem_category,
        state.profile_notes or "",
        state.conversation_history,
        readiness,
        response_style=state.turn.response_style or "full_advice",
        slots_filled=state.case.slots_filled or {},
        retrieved_info=state.retrieved_info or "",
    )
    if result:
        state.turn.coach_output = result.get("coach_output")
        state.coach_response = result.get("coach_response") or ""
    return {}
