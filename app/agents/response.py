"""Composer: blend specialist outputs into natural language final response.

This is the unified response composer that absorbs:
- dialogue_manager (decides ASK_ONE_QUESTION / RESPOND / RESPOND_ONLY)
- response_selector (picks empathy/advice text, drops low-confidence)
- formulate_response (LLM blend into final reply)

Adaptive: uses turn_mode, context_modifier, safety constraints, and
phase context to shape the response tone and content.
"""
import logging
from typing import Optional, Tuple

from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState
from app.utils import profile_is_collected
from app.agents.phase_manager import get_phase_context_for_prompt

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3

_BANNED_PHRASES_RULE = (
    "\n## BANNED phrases — NEVER use these or close variants:\n"
    "- 'I hear you' / 'I hear that'\n"
    "- 'I understand how you feel'\n"
    "- 'What would you like support with today?' / 'What would you like to focus on?'\n"
    "- 'What would you like help with?' / 'What can I help you with?'\n"
    "- 'Thanks for sharing'\n"
    "- 'That sounds difficult' / 'That sounds tough' (without specifics)\n"
    "- 'Let me suggest' / 'Here's what I recommend'\n"
    "- Starting with 'You're right' as empty validation\n"
    "These make the bot sound robotic. Reference the user's SPECIFIC words and situation instead.\n"
    "Write like a real person talking — short sentences, varied openers, no formulaic structure.\n"
)


# ---------------------------------------------------------------------------
# Component selection (absorbed from response_selector.py)
# ---------------------------------------------------------------------------

def _empathy_text(state: AppState) -> Tuple[Optional[str], float]:
    """Return (text, confidence). Prefer emotion_output when present."""
    out = state.turn.emotion_output
    if out and out.confidence >= CONFIDENCE_THRESHOLD:
        text = out.reflection or out.validation or out.gentle_reframe
        if text:
            return (text, out.confidence)
    raw = state.turn.emotion_response or state.emotion_response
    return (raw, 0.8 if raw else 0.0)


def _advice_text(state: AppState) -> Tuple[Optional[str], float]:
    """Return (text, confidence). Combine coach+growth; prepend psychoeducation/pattern."""
    # If safety says must_refuse, suppress advice entirely
    if state.turn.must_refuse and state.turn.safety_override_triggered:
        return (None, 0.0)

    coach_out = state.turn.coach_output
    growth_out = state.turn.growth_output
    coach_text = None
    coach_conf = 0.0
    growth_text = None
    growth_conf = 0.0

    if coach_out and coach_out.confidence >= CONFIDENCE_THRESHOLD:
        coach_text = coach_out.exercise_title
        if coach_out.steps:
            coach_text = (coach_text or "") + "\n" + "\n".join(f"- {s}" for s in coach_out.steps)
        coach_conf = coach_out.confidence
    if not coach_text:
        coach_text = state.turn.coach_response or state.coach_response
        if coach_text:
            coach_conf = 0.7

    if growth_out and growth_out.confidence >= CONFIDENCE_THRESHOLD:
        growth_text = growth_out.goal
        if growth_out.milestones:
            growth_text = (growth_text or "") + "\n" + "\n".join(f"- {m}" for m in growth_out.milestones)
        growth_conf = growth_out.confidence
    if not growth_text:
        growth_text = state.turn.growth_response or state.growth_response
        if growth_text:
            growth_conf = 0.7

    if coach_text and growth_text:
        combined = f"Practical step: {coach_text}\n\nLonger-term: {growth_text}"
        combined_conf = max(coach_conf, growth_conf)
    elif coach_text:
        combined = coach_text
        combined_conf = coach_conf
    elif growth_text:
        combined = growth_text
        combined_conf = growth_conf
    else:
        combined = None
        combined_conf = 0.0

    preamble_parts = []
    psych = getattr(state.turn, "psychoeducation_response", None)
    if psych:
        preamble_parts.append(psych)
    pattern = getattr(state.turn, "pattern_response", None)
    if pattern:
        preamble_parts.append(pattern)

    if preamble_parts and combined:
        combined = "\n\n".join(preamble_parts) + "\n\n" + combined
        combined_conf = max(combined_conf, 0.8)
    elif preamble_parts:
        combined = "\n\n".join(preamble_parts)
        combined_conf = 0.8

    return (combined, combined_conf)


# ---------------------------------------------------------------------------
# Dialogue action decision (absorbed from dialogue_manager.py)
# ---------------------------------------------------------------------------

def _decide_dialogue_action(state: AppState, has_substantive: bool) -> str:
    """Decide ASK_ONE_QUESTION / RESPOND_AND_OPTIONAL_QUESTION / RESPOND_ONLY."""
    readiness = getattr(state.case, "readiness_score", 0.0) or 0.0
    risk_action = (state.turn.risk_action or "").lower()

    if risk_action == "ask_safety_question":
        state.turn.follow_up_question = (
            state.turn.must_ask
            or "Are you safe right now? Do you have someone you can reach out to?"
        )
        return "ASK_ONE_QUESTION"

    if not has_substantive and readiness < 0.5:
        return "ASK_ONE_QUESTION"

    if has_substantive and readiness < 0.8:
        return "RESPOND_AND_OPTIONAL_QUESTION"

    return "RESPOND_ONLY"


def _tone_from_turn_mode(state: AppState) -> str:
    """Build tone guidance from turn_mode instead of only response_style."""
    turn_mode = state.turn.turn_mode or ""
    style = (state.turn.response_style or "empathy_light_advice").lower()
    context_mod = state.case.context_modifier or "ordinary_conflict"
    fallback = state.therapy.temporary_fallback

    turn_mode_reason = (state.turn.turn_mode_reason or "").lower()
    is_resistance = "resistance" in turn_mode_reason

    if is_resistance:
        tone = (
            "warm, validating, 2-3 sentences. The user feels overwhelmed by previous "
            "suggestions. Acknowledge that it IS a lot. Offer ONE very small, low-effort "
            "thing they could try, or simply validate that sometimes pausing is okay. "
            "Do NOT repeat previous advice or give more steps."
        )
    elif fallback or turn_mode == "empathy_containment":
        tone = (
            "warm, present, 2-4 sentences. Sit with the user's feelings in therapist "
            "process language. No advice, no steps, no exercises. End with a gentle "
            "question about their feelings, not a suggestion."
        )
    elif turn_mode == "safety_check":
        tone = (
            "calm, grounding, 2-3 sentences. Acknowledge what's happening, express "
            "concern for safety, provide one clear next step."
        )
    elif turn_mode == "psychoeducation":
        tone = (
            "explanatory, warm, 2-4 sentences. Explain the dynamic or pattern as "
            "a therapist formulation. No exercises or action steps. Optionally "
            "end with a reflective question."
        )
    elif turn_mode == "communication_coaching":
        if style == "full_advice":
            tone = (
                "Therapist-led, supportive, 3-6 sentences. Present a tailored "
                "treatment direction — briefly acknowledge the problem, explain WHY "
                "this approach fits their relational pattern, then give the concrete "
                "steps. No generic tips. Make it feel like a therapist explaining a "
                "treatment strategy, not a checklist."
            )
        else:
            tone = (
                "Lead with 2 sentences of specific empathy. Then ONE brief "
                "therapist-informed intervention suggestion (1 sentence). "
                "No rigid plans, no numbered steps."
            )
    elif turn_mode in ("trust_repair", "closeness_building"):
        tone = (
            "Warm, depth-oriented, 3-5 sentences. Focus on the emotional bond. "
            "Address the specific trust or closeness issue the user raised. "
            "Suggest one concrete repair gesture if appropriate."
        )
    elif turn_mode in ("maintenance_review", "progress_reflection"):
        tone = (
            "Reflective, affirming, 3-5 sentences. Review what has been accomplished. "
            "Highlight growth. Suggest next steps for continued progress."
        )
    elif style == "empathy_only":
        tone = (
            "warm, present, 2-4 sentences. Sit with the user's feelings. No advice."
        )
    elif style == "understanding":
        tone = (
            "explanatory, warm, 2-4 sentences. Explain the dynamic or pattern. "
            "No exercises or action steps."
        )
    elif style == "empathy_light_advice":
        tone = (
            "Lead with 2 sentences of specific empathy. Then ONE brief suggestion."
        )
    else:
        tone = (
            "Therapist-led, supportive, 3-6 sentences. Tailored treatment direction."
        )

    # Context modifier adjustments
    if context_mod == "possible_abuse":
        tone += (
            " IMPORTANT: Do not suggest couples exercises. Prioritize individual "
            "safety and validate the user's experience without assuming mutual fault."
        )
    elif context_mod == "separation_or_breakup":
        tone += " Acknowledge the loss. Do not push reconciliation."
    elif context_mod == "one_partner_unavailable":
        tone += " Frame guidance for individual reflection, not joint exercises."

    culture = state.turn.cultural_note or state.cultural_note
    if culture:
        tone += "; respect cultural context"
    if (state.case.therapy_mode or "one_person") == "two_partner":
        tone += "; balance both partner perspectives and avoid taking sides"

    return tone


def _recent_assistant_themes(conversation_history: list) -> str:
    """Return recent assistant wording so the composer can avoid loops."""
    if not conversation_history:
        return ""
    snippets = []
    for message in conversation_history[-8:]:
        if message.get("role") != "assistant":
            continue
        content = (message.get("content") or "").strip()
        if len(content) >= 40:
            snippets.append(content[:240])
    return "\n---\n".join(snippets[-3:])


def _ordering_rule(state: AppState) -> str:
    """Choose whether the reply should lead with insight or attunement."""
    turn_mode = state.turn.turn_mode or ""
    style = (state.turn.response_style or "").lower()
    coaching_ok = getattr(state.case, "coaching_eligible", False)

    if turn_mode == "psychoeducation":
        return (
            "- The user is asking for understanding: insight may lead, but use everyday "
            "language and avoid repeated clinical labels.\n"
            "- Do not add advice or scripts unless a component explicitly contains one.\n"
        )
    if style == "empathy_only" or turn_mode == "empathy_containment" or not coaching_ok:
        return (
            "- Lead with validation and specificity. Stay close to the user's words.\n"
            "- Do not use clinical pattern names, theory, homework, dialogue scripts, "
            "or action steps.\n"
            "- If an insight component is present, treat it as background and omit it unless "
            "one plain-language sentence would deepen the validation.\n"
        )
    return (
        "- Lead with empathy, then offer at most one plain-language insight if it helps, "
        "then one concrete suggestion.\n"
        "- When presenting practical advice, explain why it fits this couple first, "
        "then give one concrete thing to try, not a multi-step protocol.\n"
    )


# ---------------------------------------------------------------------------
# Main composer (unified formulate_response)
# ---------------------------------------------------------------------------

def formulate_response(state: AppState):
    """Compose final response: select components, decide dialogue action, blend with LLM.

    Enforces safety constraints: if must_refuse is set, advice is suppressed.
    Uses turn_mode and context_modifier for tone shaping.
    """
    # If the smart intake agent already set final_response, respect it.
    existing_action = (state.turn.dialogue_action or "").upper()
    if state.turn.final_response and existing_action in ("TRANSITION_TO_SUPPORT", "ASK_ONE_QUESTION"):
        logger.info("formulate_response: smart intake already set final_response, preserving (action=%s)", existing_action)
        return {"turn": state.turn.model_copy(update={"final_response": state.turn.final_response})}

    # --- Step 1: Select components ---
    empathy, empathy_conf = _empathy_text(state)
    advice, advice_conf = _advice_text(state)

    has_substantive = any([
        empathy,
        advice,
        state.retrieved_info,
        state.cultural_note,
    ])

    # --- Step 2: Decide dialogue action ---
    action = _decide_dialogue_action(state, has_substantive)
    state.turn.dialogue_action = action
    logger.info("formulate_response: action=%s has_substantive=%s", action, has_substantive)

    # --- Step 3: ASK_ONE_QUESTION path ---
    if action == "ASK_ONE_QUESTION":
        follow_up = state.turn.follow_up_question
        if follow_up:
            state.final_response = follow_up
            return {"turn": state.turn.model_copy(update={"final_response": follow_up})}

        slots = state.case.slots_filled or {}
        slots_info = ", ".join(f"{k}: {v}" for k, v in slots.items() if v) or "minimal"
        history_snippet = ""
        if state.conversation_history:
            recent = state.conversation_history[-4:]
            history_snippet = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in recent)

        system_prompt = (
            "You are a couples counseling assistant. The user has been talking but we need a bit more context.\n\n"
            f"What we know so far: {slots_info}\n"
            f"Recent conversation:\n{history_snippet}\n\n"
            + _BANNED_PHRASES_RULE +
            "Write ONE short follow-up (1-2 sentences) that:\n"
            "- References something SPECIFIC the user already said\n"
            "- Asks about something we DON'T already know\n"
            "- Sounds like a real person continuing a conversation, not a form\n"
        )
        follow_up_response = ask_model(system_prompt, user_prompt=state.text or "", model=MODEL_NAME)
        state.final_response = follow_up_response.strip()
        return {"turn": state.turn.model_copy(update={"final_response": state.turn.final_response})}

    # --- Step 4: Blend selected components ---
    tone_guidance = _tone_from_turn_mode(state)
    style = (state.turn.response_style or "empathy_light_advice").lower()
    turn_mode = state.turn.turn_mode or ""
    coaching_ok = getattr(state.case, "coaching_eligible", False)
    attunement_first = (
        style == "empathy_only"
        or turn_mode == "empathy_containment"
        or (not coaching_ok and turn_mode != "psychoeducation")
    )
    ordering_rule = _ordering_rule(state)
    recent_assistant = _recent_assistant_themes(state.conversation_history)

    follow_up = state.turn.follow_up_question if action == "RESPOND_AND_OPTIONAL_QUESTION" else None

    # Safety enforcement: suppress advice when must_refuse is active
    must_refuse = state.turn.must_refuse
    if must_refuse:
        advice = None

    if empathy is not None or advice is not None:
        empathy_str = empathy or ""
        if attunement_first:
            advice_str = ""
        else:
            advice_str = advice or ""
        follow_up_str = follow_up or ""

        phase_context = get_phase_context_for_prompt(state)
        formulation = state.case.formulation_summary or ""
        strengths = state.case.strengths_summary or ""
        focus_areas = ", ".join(state.case.focus_areas or []) or "none"
        conflict_assessment = state.case.conflict_pattern_assessment or ""
        horsemen = ", ".join(state.turn.detected_horsemen or []) or "none"
        retrieved_info = state.retrieved_info or ""
        knowledge_context = (
            "\nKnowledge retrieval (ground only, do not quote verbatim):\n"
            f"{retrieved_info}\n"
            if retrieved_info
            else ""
        )

        safety_note = ""
        if must_refuse:
            safety_note = f"\n\nSAFETY CONSTRAINT: {must_refuse}\n"

        system_prompt = (
            "## Role\n"
            "You are a couples therapist writing directly to the user. "
            "You receive pre-selected empathy and advice components. Weave them into one natural reply.\n"
            "\n## Therapy Phase Context\n"
            f"{phase_context}\n"
            f"{safety_note}"
            "\n## Background (use to ground your reply, do NOT repeat verbatim)\n"
            f"Therapist formulation: {formulation or 'not available yet'}\n"
            f"Strengths: {strengths or 'not captured yet'}\n"
            f"Focus areas: {focus_areas}\n"
            f"Conflict assessment: {conflict_assessment or 'not available yet'}\n"
            f"Four Horsemen markers: {horsemen}\n"
            f"Recent assistant replies to avoid repeating:\n{recent_assistant or 'none'}\n"
            f"{knowledge_context}"
            "\n## Components to blend\n"
            f"- Empathy: {empathy_str}\n"
            f"- Advice: {advice_str}\n"
            "\n## Style rules\n"
            f"Tone: {tone_guidance}\n"
            "- Write in the user's language.\n"
            "- Use SHORT sentences. Vary sentence length. Sound like a real person.\n"
            f"{ordering_rule}"
            "- Reference what the user ACTUALLY said. Use their own words when possible.\n"
            "- NO bullet lists, NO numbered steps, NO titles/headers.\n"
            "- Do NOT add new advice or diagnoses beyond the provided components.\n"
            "- Do NOT present advice as a formulaic exercise.\n"
            "- Do NOT repeat the same pattern label or scripted wording from recent assistant replies.\n"
            + _BANNED_PHRASES_RULE +
            "\n## Output\n"
            "Free-form paragraph(s), concise. One follow-up question only if provided below."
        )
        if follow_up_str:
            system_prompt += "\n\nEnd with this follow-up question: " + follow_up_str
        final_answer = ask_model(
            system_prompt,
            user_prompt=state.text or "",
            model=MODEL_NAME,
            history=state.conversation_history,
        )
        state.final_response = final_answer
        logger.info("formulate_response: set final_response (blend, mode=%s) len=%s",
                     state.turn.turn_mode, len(final_answer or ""))
        return {"turn": state.turn.model_copy(update={"final_response": final_answer})}

    # --- Step 5: Legacy blend fallback ---
    logger.info("formulate_response: branch legacy blend, mode=%s", state.turn.turn_mode)
    empathy_raw = state.emotion_response or ""
    advice_raw = "" if (style == "empathy_only" or must_refuse) else (state.coach_response or "")
    growth = "" if style in ("empathy_only", "understanding") or must_refuse else (state.growth_response or "")
    psychoeducation = getattr(state.turn, "psychoeducation_response", "") or ""
    pattern = getattr(state.turn, "pattern_response", "") or ""
    info = state.retrieved_info or ""
    culture_note = state.cultural_note or ""
    profile = state.profile_notes or "Not captured yet."

    has_profile_context = profile_is_collected(state.profile_notes)
    readiness = getattr(state.case, "readiness_score", None) or 0.0
    can_offer_guidance = has_profile_context and (readiness >= 0.5 or state.questions_asked >= 5)
    follow_up_raw = state.follow_up_question if not can_offer_guidance else ""

    phase_context_legacy = get_phase_context_for_prompt(state)
    ordering_rule = _ordering_rule(state)
    recent_assistant = _recent_assistant_themes(state.conversation_history)

    safety_note = ""
    if must_refuse:
        safety_note = f"\nSAFETY CONSTRAINT: {must_refuse}\n"

    system_prompt = (
        "## Role\n"
        "You are a couples therapist writing directly to the user. "
        "Blend the provided components into one natural, conversational reply.\n"
        "\n## Therapy Phase Context\n"
        f"{phase_context_legacy}\n"
        f"{safety_note}"
        "\n## Components to blend\n"
        f"- Empathy: {empathy_raw}\n"
        f"- Understanding/insight: {psychoeducation}\n"
        f"- Pattern/cycle: {pattern}\n"
        f"- Practical advice: {advice_raw}\n"
        f"- Growth tip: {growth}\n"
        f"- Extra info: {info}\n"
        f"- Cultural context: {culture_note}\n"
        f"- Profile: {profile}\n"
        "\n## Style rules\n"
        f"Tone: {tone_guidance}\n"
        "- Write in the user's language. SHORT sentences. Varied openers. Sound like a real person.\n"
        f"{ordering_rule}"
        "- Reference the user's SPECIFIC words and situation.\n"
        "- NO bullet lists, NO numbered steps, NO titles/headers.\n"
        "- Do NOT invent new content beyond what's provided.\n"
        "- Do NOT repeat the same pattern label or scripted wording from recent assistant replies.\n"
        f"Recent assistant replies to avoid repeating:\n{recent_assistant or 'none'}\n"
        + _BANNED_PHRASES_RULE +
        "\n## Output\n"
        "Free-form paragraph(s), concise and flowing."
    )
    if follow_up_raw:
        system_prompt += "\n\nInclude this follow-up as the closing sentence: " + follow_up_raw

    final_answer = ask_model(
        system_prompt,
        user_prompt=state.text or "",
        model=MODEL_NAME,
        history=state.conversation_history,
    )
    state.final_response = final_answer
    logger.info("formulate_response: set final_response (legacy blend, mode=%s) len=%s",
                 state.turn.turn_mode, len(final_answer or ""))
    return {"turn": state.turn.model_copy(update={"final_response": final_answer})}
