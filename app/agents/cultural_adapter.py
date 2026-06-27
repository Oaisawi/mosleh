"""Cultural Adapter: cultural/religious alignment of advice. No safety decisions."""
import logging

from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.logutil import ctx_from_state
from app.models import AppState

logger = logging.getLogger(__name__)


def cultural_adapter(state: AppState):
    """
    Inputs: user_culture, gender, topic, boundaries from profile.
    Output: cultural_note + optional phrasing_guidelines for composer.
    """
    culture = state.profile.culture or state.user_culture
    if not culture:
        logger.info("cultural_adapter %s skip reason=no_culture", ctx_from_state(state))
        state.turn.cultural_note = None
        state.turn.phrasing_guidelines = None
        return {}
    logger.info(
        "cultural_adapter %s llm_path advice_so_far_len=%s",
        ctx_from_state(state),
        len((state.coach_response or "") + (state.growth_response or "")),
    )
    advice_so_far = ""
    if state.coach_response:
        advice_so_far += state.coach_response + "\n"
    if state.growth_response:
        advice_so_far += state.growth_response
    system_prompt = (
        "## Role\n"
        f"You are the cultural/religious alignment reviewer for the couples counseling assistant. You are familiar with {culture} culture.\n"
        "\n## Background\n"
        "Advice has already been generated. You review it for alignment with the user's stated culture (e.g. from profile). You do not make safety or clinical decisions. Your output is used by the composer.\n"
        "\n## Tasks\n"
        "Review the advice text and output a short cultural_note and optional phrasing_guidelines for the composer.\n"
        "\n## Do\n"
        "Stay brief. Suggest modifications that respect the stated culture. Output exactly two labeled lines.\n"
        "\n## Do not\n"
        "Do not override safety or clinical content. Do not write long paragraphs. Do not change the meaning of the advice.\n"
        "\n## Output format\n"
        "Return exactly two lines:\n"
        "Cultural_note: <one or two sentences of culturally appropriate phrasing or context>\n"
        "Phrasing_guidelines: <brief tone/language guidance for the composer>"
    )
    raw = ask_model(
        system_prompt,
        user_prompt=advice_so_far or state.text or "",
        model=MODEL_NAME,
        history=state.conversation_history,
    )
    cultural_note = None
    phrasing_guidelines = None
    for line in raw.splitlines():
        lower = line.lower()
        if lower.startswith("cultural_note"):
            cultural_note = line.split(":", 1)[-1].strip()
        elif lower.startswith("phrasing_guidelines"):
            phrasing_guidelines = line.split(":", 1)[-1].strip()
    state.turn.cultural_note = cultural_note or raw.strip()
    state.turn.phrasing_guidelines = phrasing_guidelines
    return {}


def therapy_specialist_agent(state: AppState):
    """Legacy: single node that does cultural adapter only (for backward compat)."""
    cultural_adapter(state)
    return {}
