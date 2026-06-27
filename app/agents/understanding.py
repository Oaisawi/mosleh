"""Combined understanding: one LLM call for emotion, sentiment, and problem category."""
import logging
from typing import Optional

from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState

logger = logging.getLogger(__name__)

PROBLEM_CATEGORIES = [
    "Communication",
    "Financial Stress",
    "Child Related Conflicts",
    "Emotional Distance",
    "Family Interference",
    "Intimacy & Affection",
    "Trust Issues",
    "Cultural/Value Differences",
    "Other",
]
CATEGORY_LIST = ", ".join(PROBLEM_CATEGORIES)

# Keyword hints: if user text contains these, prefer this category (overrides LLM "Other" when obvious)
CATEGORY_KEYWORD_HINTS = [
    ("communication", "communicat", "talking", "listen", "argue", "arguing"),
    ("financial", "money", "finances", "debt", "spending", "budget"),
    ("child", "children", "kids", "parenting", "custody"),
    ("emotional distance", "distance", "disconnect", "drift apart", "growing apart"),
    ("family", "in-law", "in-laws", "parents", "interference", "meddling"),
    ("intimacy", "affection", "physical", "sex", "closeness"),
    ("trust", "trust issues", "cheat", "lying", "honesty", "transparency"),
    ("cultural", "culture", "values", "beliefs", "religion", "tradition"),
]
CATEGORY_FROM_KEYWORDS = [
    "Communication",
    "Financial Stress",
    "Child Related Conflicts",
    "Emotional Distance",
    "Family Interference",
    "Intimacy & Affection",
    "Trust Issues",
    "Cultural/Value Differences",
]


def _category_from_keywords(text: str) -> Optional[str]:
    """If user text clearly mentions a problem domain, return the matching category."""
    if not text or len(text.strip()) < 5:
        return None
    t = text.lower()
    for keywords, category in zip(CATEGORY_KEYWORD_HINTS, CATEGORY_FROM_KEYWORDS):
        if any(kw in t for kw in keywords):
            return category
    return None


def combined_understanding(state: AppState) -> dict:
    """
    Single LLM call to set emotion, sentiment, and problem_category.
    Skips LLM when category is already set and we don't need emotion for this turn.
    """
    run_emotion = getattr(state.turn, "run_emotion", False)
    run_coach = getattr(state.turn, "run_coach", False)
    run_growth = getattr(state.turn, "run_growth", False)
    need_any = run_emotion or run_coach or run_growth
    if not need_any:
        logger.info("combined_understanding: skipping (need_any=False)")
        return {}

    existing_category = (state.case.problem_category or "").strip()
    has_category = (
        len(existing_category) > 0
        and existing_category.lower() not in {"not enough", "not_enough", "notenough", "insufficient"}
    )
    need_emotion_for_turn = run_emotion

    recent_blob = (
        " ".join((state.turn.recent_user_messages or state.recent_user_messages or [])[-5:])
        if (state.turn.recent_user_messages or state.recent_user_messages)
        else (state.turn.text or state.text or "")
    )
    profile_blob = (state.profile.profile_notes or state.profile_notes or "").strip()
    if len((recent_blob or "").strip()) < 20:
        if len(profile_blob) >= 20:
            recent_blob = profile_blob
        else:
            logger.info("combined_understanding: blob too short, setting not enough")
            state.turn.emotion = "not enough"
            state.turn.sentiment = ""
            if not has_category:
                state.case.problem_category = "not enough"
            return {}

    # Skip LLM: we have category and don't need emotion this turn; set lightweight emotion from triage
    if has_category and not need_emotion_for_turn:
        logger.info("combined_understanding: skip LLM (has_category, no need_emotion_for_turn)")
        intensity = getattr(state.turn, "emotional_intensity", 0.0) or 0.0
        if intensity >= 0.6:
            state.turn.emotion = "stressed"
            state.turn.sentiment = "negative"
        elif intensity >= 0.3:
            state.turn.emotion = "concerned"
            state.turn.sentiment = "negative"
        else:
            state.turn.emotion = "neutral"
            state.turn.sentiment = "neutral"
        return {}

    system_prompt = (
        "## Role\n"
        "You are the analyst for a couples counseling app. You perform a single call that sets emotion, sentiment, and problem category.\n"
        "\n## Background\n"
        "Your output drives specialist routing (emotion/coach/growth) and RAG. It must be consistent and parseable; downstream code expects exactly three labeled lines.\n"
        "\n## Tasks\n"
        "Analyze the user's message (and recent context) and output exactly three lines: primary emotion, sentiment, and one problem category from the list.\n"
        "\n## Do\n"
        "Use the exact category list below. When the user explicitly mentions a problem area (e.g. communication, trust, money, intimacy, children, family, emotional distance), choose the matching category. Use 'Other' only when no category fits. Use 'not enough' for Emotion and Problem_category only when context is truly insufficient to classify.\n"
        "\n## Do not\n"
        "Do not invent categories. Do not output prose or extra lines. Do not output more than three lines.\n"
        "\n## Output format\n"
        "Output exactly three lines in this format:\n"
        "Emotion: <primary emotion, e.g. happy, sad, angry, fearful, neutral, stressed, overwhelmed>\n"
        "Sentiment: <positive or negative or neutral>\n"
        f"Problem_category: <exactly one of: {CATEGORY_LIST}>\n"
        "Example: Emotion: stressed\nSentiment: negative\nProblem_category: Communication\n"
        "If there is not enough context to classify, use: Emotion: not enough, Sentiment: (blank), Problem_category: not enough"
    )
    logger.info("combined_understanding: calling LLM (recent_blob len=%s)", len(recent_blob or ""))
    raw = ask_model(
        system_prompt,
        user_prompt=recent_blob,
        model=MODEL_NAME,
        history=state.conversation_history,
    )
    emotion = sentiment = category = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("emotion"):
            emotion = line.split(":", 1)[-1].strip()
        elif lower.startswith("sentiment"):
            sentiment = line.split(":", 1)[-1].strip()
        elif lower.startswith("problem_category") or lower.startswith("problem category"):
            category = line.split(":", 1)[-1].strip()

    state.turn.emotion = emotion or "not enough"
    state.turn.sentiment = sentiment or ""
    keyword_category = _category_from_keywords(recent_blob or "")

    if category:
        normalized = category.strip().lower()
        if normalized in {"not enough", "not_enough", "notenough", "insufficient"}:
            if len((recent_blob or "").strip()) >= 20 or len(profile_blob) >= 20:
                state.case.problem_category = keyword_category or "Other"
            else:
                state.case.problem_category = "not enough"
        else:
            # Match to list (case-insensitive)
            for c in PROBLEM_CATEGORIES:
                if c.lower() == normalized:
                    state.case.problem_category = c
                    break
            else:
                # LLM returned something not in list; if it's "other"/"Other" and we have keyword hint, use hint
                if normalized == "other" and keyword_category:
                    state.case.problem_category = keyword_category
                else:
                    state.case.problem_category = category.strip()
    elif not has_category:
        state.case.problem_category = keyword_category or "not enough"

    logger.info("combined_understanding: set emotion=%s sentiment=%s problem_category=%s", state.turn.emotion, state.turn.sentiment, state.case.problem_category)
    return {}
