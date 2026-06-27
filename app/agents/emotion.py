"""Emotion detection and empathetic response with structured output and confidence."""
import re
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState, EmotionOutput


def emotion_detection(state: AppState):
    """Detect the primary emotion and sentiment of the user's message."""
    if not state.text or not (state.need_emotion or state.need_coach or state.need_growth):
        return {}
    recent_blob = (
        " ".join(state.recent_user_messages[-5:])
        if state.recent_user_messages
        else state.text
    )
    if len(recent_blob.strip()) < 20:
        state.emotion = "not enough"
        state.sentiment = ""
        return {"emotion": state.emotion, "sentiment": state.sentiment}
    system_prompt = (
        "## Role\n"
        "You are the emotion analysis sub-step for a couples counseling app. You identify primary emotion and sentiment only.\n"
        "\n## Background\n"
        "Your output is not shown directly to the user; it feeds the emotion response agent. It must be parseable as exactly two fields.\n"
        "\n## Tasks\n"
        "From the user's message (and recent context), identify the primary emotion and the sentiment.\n"
        "\n## Do\n"
        "Use a small fixed emotion set: happy, sad, angry, fearful, neutral, stressed, overwhelmed. Output exactly two fields in the format below.\n"
        "\n## Do not\n"
        "Do not give advice or respond to the user. Do not output long text or prose.\n"
        "\n## Output format\n"
        "Respond in exactly this format: Emotion: <emotion>, Sentiment: <sentiment>"
    )
    analysis = ask_model(
        system_prompt,
        user_prompt=recent_blob,
        model=MODEL_NAME,
        history=state.conversation_history,
    )
    parts = analysis.split(",")
    if len(parts) >= 2:
        emotion = parts[0].split(":")[-1].strip()
        sentiment = parts[1].split(":")[-1].strip()
    else:
        emotion = analysis.strip()
        sentiment = ""
    state.emotion = emotion
    state.sentiment = sentiment
    return {"emotion": state.emotion, "sentiment": state.sentiment}


def _parse_emotion_output(raw: str, fallback_confidence: float = 0.8) -> EmotionOutput:
    """Parse LLM response into EmotionOutput. Expects labeled lines or fallback to single reflection."""
    reflection = validation = gentle_reframe = None
    confidence = fallback_confidence
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("reflection"):
            reflection = line.split(":", 1)[-1].strip()
        elif lower.startswith("validation"):
            validation = line.split(":", 1)[-1].strip()
        elif lower.startswith("gentle_reframe") or lower.startswith("gentle reframe"):
            gentle_reframe = line.split(":", 1)[-1].strip()
        elif lower.startswith("confidence"):
            try:
                confidence = float(re.search(r"[\d.]+", line).group(0))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, AttributeError):
                pass
    if not reflection and raw.strip():
        reflection = raw.strip()
    return EmotionOutput(
        reflection=reflection,
        validation=validation,
        gentle_reframe=gentle_reframe,
        confidence=confidence,
    )


def emotion_agent_compute(
    text: str,
    emotion: str,
    conversation_history: list,
) -> dict:
    """
    Pure compute: LLM call + parse. Returns dict with emotion_output, emotion_response.
    Used by parallel_specialists and by emotion_agent.
    """
    if not text or not emotion or (emotion or "").lower() == "not enough":
        return {}
    system_prompt = (
        "## Role\n"
        "You are the emotion specialist for the couples counseling assistant. You provide only empathetic reflection and validation.\n"
        "\n## Background\n"
        "The user has shared feelings. Your output may be shown as the 'empathy' part of the final reply. No advice or steps.\n"
        "\n## Tasks\n"
        "Produce reflection, validation, optional gentle reframe, and a confidence score (0.0 to 1.0). Do not give advice or steps.\n"
        "\n## Do\n"
        "Respond in the user's language. Keep each line short (one sentence). Reference the user's SPECIFIC situation (not generic feelings). Output confidence as a number between 0.0 and 1.0.\n"
        "When the user's words point to shame, dignity, or self-worth pain (for example feeling small, stupid, not worth effort, or like they are begging), name that gently without diagnosing.\n"
        "\n## Do not\n"
        "Do not give advice, exercises, or questions. Do not diagnose or minimize their experience.\n"
        "NEVER start with 'I hear you', 'I hear that', or 'I understand'. These are overused and robotic.\n"
        "NEVER use generic filler like 'That sounds difficult' without referencing what specifically is difficult.\n"
        "\n## Output format\n"
        f"The user feels {emotion}. Return exactly these four lines:\n"
        "Reflection: <one sentence that references SPECIFIC details from what the user said — not generic>\n"
        "Validation: <one sentence validating their experience with specifics>\n"
        "Gentle_reframe: <optional gentle reframe or hope, or leave blank>\n"
        "Confidence: <0.0 to 1.0>\n"
        "\nGood examples (notice: specific, varied openers):\n"
        "Reflection: Going weeks without a real conversation with your wife — that kind of silence can feel really lonely.\n"
        "Reflection: Watching the distance grow between you two when you used to be so close — that's a painful shift.\n"
        "Reflection: Feeling like you're drifting apart after years together makes sense given what you're describing.\n"
        "\nBad examples (NEVER do this):\n"
        "Reflection: I hear you — that sounds really hard.\n"
        "Reflection: I understand how you feel.\n"
        "Reflection: That must be difficult for you.\n"
    )
    raw = ask_model(
        system_prompt,
        user_prompt=text,
        model=MODEL_NAME,
        history=conversation_history or [],
    )
    out = _parse_emotion_output(raw)
    response_text = out.reflection or out.validation or (raw.strip() if raw else "")
    return {
        "emotion_output": out,
        "emotion_response": response_text,
    }


def emotion_agent(state: AppState):
    """Generate an empathetic response with structured output (reflection, validation, gentle_reframe, confidence)."""
    if (
        not state.text
        or not state.emotion
        or not state.need_emotion
        or (state.emotion or "").lower() == "not enough"
    ):
        return {}
    result = emotion_agent_compute(
        state.text,
        state.emotion,
        state.conversation_history,
    )
    if result:
        state.turn.emotion_output = result.get("emotion_output")
        state.emotion_response = result.get("emotion_response") or ""
    return {}
