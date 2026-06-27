"""Smart context-aware intake agent.

Unified intake that handles EVERYTHING from the first greeting onward:
- Greetings: warm welcome + opening question (no separate smalltalk agent)
- Short answers: extracts slot info, asks next question
- Distress: validates emotions first
- Help-now: transitions to support immediately
- Also builds profile summary (absorbs run_user_info_collection)
"""
import logging
from typing import Dict, List, Optional

from app.config import (
    MODEL_NAME,
    GREETINGS,
    SLOTS_GENERIC,
    SLOTS_THERAPY_OPTIONAL,
    SLOT_QUESTIONS,
    SLOTS_REQUIRED_BY_CATEGORY,
    THERAPY_PHASES,
)
from app.llm.providers import ask_model
from app.models import AppState
from app.agents.phase_manager import get_phase_context_for_prompt

logger = logging.getLogger(__name__)

# Cues that signal user is emotionally distressed
DISTRESS_CUES = {
    "frustrated", "angry", "upset", "overwhelmed", "can't take", "tired of",
    "hopeless", "exhausted", "falling apart", "crying", "scared", "miserable",
    "fed up", "giving up", "breaking down", "had enough", "so done",
    "can't anymore", "lost", "helpless", "furious", "devastated",
}

# Cues that the user wants concrete help now, not more questions
HELP_NOW_CUES = {
    "help me", "what should i do", "what do i do", "i don't know what to do",
    "i need help", "what is doable", "give me something", "suggest",
    "any tips", "what can i try", "how do i", "what should",
    "step by step", "step-by-step", "concrete", "practical",
}

# Banned phrases that make the bot sound robotic / context-unaware
_BANNED_PHRASES_RULE = (
    "BANNED PHRASES — NEVER use any of these or close variants:\n"
    "- 'What would you like support with today?'\n"
    "- 'What would you like to focus on?'\n"
    "- 'What would you like help with?'\n"
    "- 'What can I help you with?'\n"
    "- 'How can I assist you?'\n"
    "- 'What brings you here today?'\n"
    "- 'Thanks for sharing'\n"
    "- 'I hear you'\n"
    "- 'I understand how you feel'\n"
    "These sound like the conversation is starting over. The user ALREADY told you their problem.\n"
)


def _required_slots(category: Optional[str]) -> List[str]:
    if not category:
        return SLOTS_GENERIC
    return SLOTS_REQUIRED_BY_CATEGORY.get(category, SLOTS_GENERIC)


def _therapy_optional_slots() -> List[str]:
    return list(SLOTS_THERAPY_OPTIONAL)


def _missing_slots(slots: Dict[str, str], category: Optional[str]) -> List[str]:
    required = _required_slots(category)
    return [s for s in required if not slots.get(s)]


def _slots_summary(slots: Dict[str, str], category: Optional[str]) -> str:
    required = _required_slots(category)
    optional = _therapy_optional_slots()
    lines = []
    for s in required + optional:
        val = slots.get(s, "")
        status = f"FILLED: {val}" if val else ("MISSING" if s in required else "OPTIONAL")
        label = SLOT_QUESTIONS.get(s, s)
        lines.append(f"  - {s} ({status}) — question hint: \"{label}\"")
    return "\n".join(lines)


def _detect_distress(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in DISTRESS_CUES)


def _detect_help_now(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in HELP_NOW_CUES)


def _is_greeting(text: str) -> bool:
    """Check if the message is a pure greeting (hello, hi, etc.)."""
    t = (text or "").lower().strip()
    if not t:
        return True
    # Pure greeting: entire message is a greeting word or very short
    words = t.split()
    if len(words) <= 3 and any(g in t for g in GREETINGS):
        return True
    return False


def _readiness_score(slots: Dict[str, str], category: Optional[str], questions_asked: int) -> float:
    """Calculate how ready we are to transition from intake to support.
    0.0 = just started, 1.0 = fully ready."""
    required = _required_slots(category)
    if not required:
        return 1.0
    filled = sum(1 for s in required if slots.get(s))
    slot_score = filled / len(required)
    # Also give credit for questions asked (even if slots weren't perfectly extracted)
    question_score = min(questions_asked / 6.0, 1.0)
    return max(slot_score, question_score * 0.8)


def _extract_slots_from_reply(
    user_text: str,
    history: List[Dict[str, str]],
    current_slots: Dict[str, str],
    category: Optional[str],
) -> Dict[str, str]:
    """Use LLM to extract slot values from user reply and merge into current_slots."""
    required = _required_slots(category)
    optional = _therapy_optional_slots()
    all_slots = required + optional
    slots_blob = "\n".join(f"- {k}: {current_slots.get(k, '')}" for k in all_slots)

    recent_context = ""
    if history:
        recent = history[-4:]
        recent_context = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in recent)

    system_prompt = (
        "You are the intake fact-extractor for a couples counseling app.\n"
        "Extract facts from the user's latest message AND the recent conversation context.\n\n"
        f"Current slots:\n{slots_blob}\n\n"
        f"Recent conversation:\n{recent_context}\n\n"
        "Slot descriptions:\n"
        "- situation_summary: The main issue or concern (e.g. 'communication problems', 'trust issues', 'drifting apart')\n"
        "- who_involved: Who is affected (e.g. 'user and wife', 'user and partner')\n"
        "- timeframe: How long this has been going on (e.g. '2 months', 'recently', 'a few weeks')\n"
        "- what_tried: What they've attempted so far (e.g. 'nothing yet', 'tried talking', 'couples therapy')\n"
        "- desired_outcome: What they want to achieve (e.g. 'better communication', 'reconnect', 'practical steps')\n"
        "- constraints: Anything to keep in mind (e.g. 'children involved', 'long distance', 'privacy concerns')\n\n"
        "- relationship_length: duration of relationship (e.g. '7 years')\n"
        "- how_met: brief story of relationship beginning\n"
        "- major_transitions: major life transitions that impacted relationship\n"
        "- conflict_triggers: recurring conflict trigger topics/situations\n"
        "- how_arguments_end: typical ending pattern of arguments\n"
        "- repair_attempts: what happens after conflict to reconnect\n"
        "- relationship_strengths: existing strengths in the relationship\n"
        "- success_criteria: how user will know therapy is helping\n"
        "- partner_perspective: user's best understanding of partner's perspective\n\n"
        "Rules:\n"
        "- Be LENIENT: short answers like 'communication', 'nothing', 'me and my wife' ARE valid slot values\n"
        "- Look at the assistant's QUESTION and the user's ANSWER together to infer which slot it fills\n"
        "- Example: If assistant asked 'What have you tried?' and user said 'nothing yet' → what_tried: nothing yet\n"
        "- Example: If user said 'communication' in response to 'What's your main concern?' → situation_summary: communication issues\n"
        "- One line per slot that has NEW info, format: SLOT_NAME: value\n"
        "- Valid slot names: " + ", ".join(all_slots) + "\n"
        "- Do NOT repeat existing slot values. Only output NEW or UPDATED info.\n"
        "- If truly no new info, output: NO_NEW_SLOTS"
    )
    raw = ask_model(system_prompt, user_prompt=user_text, history=history[-4:] if history else [])
    updated = dict(current_slots)
    for line in raw.splitlines():
        if ":" in line and "NO_NEW_SLOTS" not in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key in required and val:
                updated[key] = val
    return updated


def smart_intake_agent(state: AppState):
    """Context-aware intake node — handles greetings, intake questions, and transitions.

    Behaviour:
    1. If pure greeting (no slots, no history): welcome warmly + ask opening question.
    2. Otherwise extract slot info from the user's latest message.
    3. Compute readiness. If ready enough, signal transition.
    4. If user is distressed or asking for help: validate + transition.
    5. Otherwise, ask ONE natural, context-aware question for the most useful missing slot.
    """
    text = state.turn.text or ""
    history = state.conversation_history or []
    slots = state.case.slots_filled or {}
    category = state.case.problem_category
    questions_asked = state.case.questions_asked or 0
    therapy_mode = (state.case.therapy_mode or "one_person").lower()
    speaker = state.turn.active_speaker or "A"
    is_greeting_msg = _is_greeting(text)

    # Phase context for prompts
    phase_context = get_phase_context_for_prompt(state)

    # --- Step 1: Handle pure greetings ---
    if is_greeting_msg and questions_asked == 0 and not any(slots.values()):
        mode_label = "two-partner therapy" if therapy_mode == "two_partner" else "one-person therapy"
        state.turn.final_response = (
            "Hey, welcome — I'm glad you reached out. "
            "I'm Mosleh AI, your couples therapist assistant. "
            f"We'll work through this in a structured {mode_label} process. "
            "What's been going on between you and your partner?"
        )
        state.turn.dialogue_action = "ASK_ONE_QUESTION"
        state.case.questions_asked = 1
        state.case.readiness_score = 0.0
        logger.info("smart_intake: greeting — static welcome, questions=1")
        return {}

    # --- Step 2: Extract slots from latest message ---
    if text.strip() and not is_greeting_msg:
        slots = _extract_slots_from_reply(text, history, slots, category)
        state.case.slots_filled = slots

    # --- Step 3: Compute readiness ---
    readiness = _readiness_score(slots, category, questions_asked)
    state.case.readiness_score = readiness

    is_distressed = _detect_distress(text)
    wants_help_now = _detect_help_now(text)
    missing = _missing_slots(slots, category)

    # --- Step 4: Decide action ---
    # Transition to support when enough context exists OR the user needs
    # help now. Phase progression itself is managed by phase_manager;
    # intake just signals that slot collection is sufficient.
    should_transition = (
        readiness >= 0.6
        or questions_asked >= 7
        or (wants_help_now and readiness >= 0.2)
        or (is_distressed and readiness >= 0.2)
        or not missing
    )

    if should_transition:
        state.case.intake_completed = True
        state.turn.turn_type = "intake_to_support"

    # --- Step 5: Generate response ---
    slots_context = _slots_summary(slots, category)
    recent_history = history[-6:] if history else []
    history_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in recent_history)

    if should_transition:
        actually_advanced = (
            state.therapy.phase_transition_decision == "advance"
            or state.therapy.current_phase > 1
        )
        if actually_advanced:
            transition_framing = (
                "The therapy has progressed to a new phase. "
                "Acknowledge that enough context has been gathered and explain "
                "that you are now moving into deeper exploration and intervention work."
            )
        else:
            transition_framing = (
                "Enough context has been gathered for now. "
                "Signal that you have a good understanding of the situation and "
                "will begin working on understanding the dynamics and building a plan. "
                "Do NOT claim a phase transition or imply the session has advanced to "
                "a different stage — remain in the current assessment framing while "
                "beginning support work."
            )
        system_prompt = (
            "You are Mosleh AI, a therapist-style couples counseling assistant.\n\n"
            f"## Therapy Phase Context\n{phase_context}\n\n"
            f"{transition_framing}\n\n"
            "Context collected:\n" + slots_context + "\n\n"
            "Recent conversation:\n" + history_text + "\n\n"
            + _BANNED_PHRASES_RULE + "\n"
            "Rules:\n"
            "- Reference their SPECIFIC situation (e.g. 'You mentioned you and your wife have been talking less for about 2 months')\n"
            "- Summarize what you've understood about their situation\n"
            "- If the user is distressed, validate their feelings FIRST with specifics\n"
            "- Do NOT ask another question — GIVE them something actionable or transitional\n"
            "- Do NOT use vague phrases like 'What would you like to focus on?' — YOU decide based on what they told you\n"
            "- Keep it to 2-3 sentences max\n"
        )
    elif is_distressed:
        system_prompt = (
            "You are Mosleh AI, a therapist-style couples counseling assistant.\n\n"
            f"## Therapy Phase Context\n{phase_context}\n\n"
            "The user seems emotionally distressed. PRIORITISE emotional validation.\n\n"
            "What we know:\n" + slots_context + "\n\n"
            "Recent conversation:\n" + history_text + "\n\n"
            + _BANNED_PHRASES_RULE + "\n"
            "Rules:\n"
            "- Reference their SPECIFIC situation — not generic empathy\n"
            "- First validate their feeling in 1 specific sentence (mention WHAT is hard, not just 'that sounds hard')\n"
            "- Then optionally ask ONE gentle, relevant question if we need critical info\n"
            "- Do NOT use clinical language or ask 'How is this affecting you?'\n"
            "- Do NOT re-ask anything already answered\n"
            "- Keep it warm, brief (2 sentences max)\n"
        )
    else:
        missing_str = ", ".join(missing[:3]) if missing else "none"
        optional_targets = [s for s in _therapy_optional_slots() if not slots.get(s)]
        optional_str = ", ".join(optional_targets[:2]) if optional_targets else "none"
        system_prompt = (
            "You are Mosleh AI, a therapist-style couples counseling assistant.\n\n"
            f"## Therapy Phase Context\n{phase_context}\n\n"
            "You are in the Assessment phase — building a safe therapeutic relationship and understanding the couple's situation.\n\n"
            "What we know so far:\n" + slots_context + "\n\n"
            "Recent conversation:\n" + history_text + "\n\n"
            "Missing required info (pick the MOST useful one): " + missing_str + "\n"
            "Useful optional therapy context (ask only when natural): " + optional_str + "\n"
            f"Current speaker in this session: Partner {speaker}\n\n"
            + _BANNED_PHRASES_RULE + "\n"
            "Rules:\n"
            "- Ask exactly ONE question — natural, conversational, NOT robotic\n"
            "- NEVER re-ask something already answered (check the slots and history above!)\n"
            "- Reference what the user said to show you're listening\n"
            "- Build safety and trust — be warm and non-judgmental\n"
            "- If they said something vague, gently ask to elaborate on THAT specific thing\n"
            "- Keep it to 1-2 sentences max\n"
            "- Do NOT use clinical/template language\n"
        )

    response = ask_model(system_prompt, user_prompt=text, history=recent_history)

    # Clean up: ensure single question
    if not should_transition and response.count("?") > 1:
        parts = response.split("?")
        response = parts[0] + "?"
        if len(parts[0]) < 20 and len(parts) > 1:
            response = parts[0] + "? " + parts[1] + "?"

    state.turn.final_response = response.strip()
    state.turn.dialogue_action = "ASK_ONE_QUESTION" if not should_transition else "TRANSITION_TO_SUPPORT"
    state.case.questions_asked = questions_asked + 1

    # --- Step 6: Build profile summary (absorbs run_user_info_collection) ---
    filled_summary = " | ".join(f"{k}: {v}" for k, v in slots.items() if v)
    if filled_summary:
        state.profile.profile_notes = filled_summary

    logger.info(
        "smart_intake: readiness=%.2f questions=%d distressed=%s wants_help=%s transition=%s missing=%s",
        readiness, questions_asked + 1, is_distressed, wants_help_now, should_transition, missing[:2],
    )

    return {}
