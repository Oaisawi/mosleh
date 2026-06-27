"""Growth agent: long-term relationship growth goals with structured output and confidence.

Situation-aware: receives filled slots to tailor goals to the user's reality.
No-repeat: avoids suggesting goals that overlap with prior advice in the conversation.
"""
import re
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState, GrowthOutput
from app.utils import profile_is_collected


def _parse_growth_output(raw: str, fallback_confidence: float = 0.75) -> GrowthOutput:
    """Parse LLM response into GrowthOutput."""
    goal = smart_breakdown = None
    milestones = []
    obstacles = []
    confidence = fallback_confidence
    for line in raw.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lower = line_stripped.lower()
        if lower.startswith("goal"):
            goal = line_stripped.split(":", 1)[-1].strip()
        elif lower.startswith("smart_breakdown") or lower.startswith("breakdown"):
            smart_breakdown = line_stripped.split(":", 1)[-1].strip()
        elif lower.startswith("milestone"):
            val = line_stripped.split(":", 1)[-1].strip()
            if val:
                milestones.append(val)
        elif lower.startswith("obstacle"):
            val = line_stripped.split(":", 1)[-1].strip()
            if val:
                obstacles.append(val)
        elif lower.startswith("confidence"):
            try:
                confidence = float(re.search(r"[\d.]+", line_stripped).group(0))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, AttributeError):
                pass
    if not goal and raw.strip():
        goal = raw.strip()[:80]
    return GrowthOutput(
        goal=goal,
        smart_breakdown=smart_breakdown,
        milestones=milestones,
        obstacles=obstacles,
        confidence=confidence,
    )


def _build_situation_context(slots_filled: dict) -> str:
    """Build a concise situation summary from filled slots."""
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
            if len(content) > 50:
                prior.append(content[:200])
    if not prior:
        return ""
    return "\n---\n".join(prior[-3:])


def growth_agent_compute(
    text: str,
    problem_category: str,
    profile_notes: str,
    conversation_history: list,
    readiness_score: float = 0.0,
    slots_filled: dict = None,
    retrieved_info: str = "",
) -> dict:
    """
    Pure compute: LLM call + parse. Returns dict with growth_output, growth_response.
    Used by parallel_specialists and by growth_agent.
    """
    if not profile_is_collected(profile_notes) or readiness_score < 0.5:
        return {}
    if (
        not text
        or (problem_category or "").lower() == "not enough"
    ):
        return {}

    category = problem_category or "Other"
    situation_context = _build_situation_context(slots_filled or {})
    prior_advice = _extract_prior_advice(conversation_history)
    knowledge_context = ""
    if retrieved_info.strip():
        knowledge_context = (
            "\n## Retrieved knowledge context\n"
            "Use this only if relevant to ground the goal. Do not quote it "
            "verbatim, and do not mention retrieval or sources in the answer.\n"
            f"{retrieved_info.strip()}\n"
        )

    no_repeat_rule = ""
    if prior_advice:
        no_repeat_rule = (
            "\n## CRITICAL: Do NOT repeat prior advice\n"
            "The following was ALREADY discussed. Propose something DIFFERENT or deeper.\n"
            f"Prior advice:\n{prior_advice}\n"
            "---\n"
        )

    system_prompt = (
        "## Role\n"
        f"You are the long-term growth specialist for couples counseling ({category}). "
        "You help the user build a realistic path forward tailored to their specific situation.\n"
        "\n## User's Situation\n"
        f"{situation_context or 'See conversation history.'}\n"
        f"Profile: {profile_notes}\n"
        f"{knowledge_context}"
        "\n## Tasks\n"
        "Propose one SMART goal that is TAILORED to this couple's reality — "
        "what they've tried, their constraints, their partner's personality. "
        "Include a brief breakdown, 1-2 milestones, and one potential obstacle specific to them.\n"
        "\n## Do\n"
        "- Reference their specific situation (what failed, what they want, partner traits)\n"
        "- Make milestones concrete and measurable for THEIR case\n"
        "- Name an obstacle that is realistic for THEM (not generic)\n"
        "- Keep the goal achievable and specific\n"
        "\n## Do not\n"
        "- Do not give generic goals ('improve communication' — too vague)\n"
        "- Do not give crisis or safety advice\n"
        "- Do not promise outcomes\n"
        "- Do not output long prose\n"
        + no_repeat_rule +
        "\n## Output format\n"
        "Use this exact format:\n"
        "Goal: <one SMART goal tailored to their situation>\n"
        "Smart_breakdown: <brief breakdown specific to their reality>\n"
        "Milestone 1: <first concrete milestone>\n"
        "Milestone 2: <optional second milestone>\n"
        "Obstacle: <one realistic obstacle for THIS couple>\n"
        "Confidence: <0.0 to 1.0>\n"
    )
    raw = ask_model(
        system_prompt,
        user_prompt=text,
        model=MODEL_NAME,
        history=conversation_history or [],
    )
    out = _parse_growth_output(raw)
    response_text = out.goal or ""
    if out.milestones:
        response_text += "\n" + "\n".join(f"- {m}" for m in out.milestones)
    return {
        "growth_output": out,
        "growth_response": response_text,
    }


def growth_agent(state: AppState):
    """Suggest growth strategy tailored to the user's specific situation."""
    category = state.problem_category
    readiness = getattr(state.case, "readiness_score", 0.0) or 0.0
    if not profile_is_collected(state.profile_notes) or readiness < 0.5:
        return {}
    if (
        not state.text
        or not state.need_growth
        or (category and category.lower() == "not enough")
    ):
        return {}
    result = growth_agent_compute(
        state.text,
        state.problem_category or "Other",
        state.profile_notes or "",
        state.conversation_history,
        readiness,
        slots_filled=state.case.slots_filled or {},
        retrieved_info=state.retrieved_info or "",
    )
    if result:
        state.turn.growth_output = result.get("growth_output")
        state.growth_response = result.get("growth_response") or ""
    return {}
