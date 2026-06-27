"""Self-evaluation hooks: lightweight post-check on final response. Run only when risk medium+ or low confidence."""
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState

# Set to True to enable self-eval (adds one LLM call when conditions hold). Default False for speed.
SELF_EVAL_ENABLED = False


def _should_run_self_eval(state: AppState) -> bool:
    """Run only when risk medium+, or low classification confidence, or user confusion cues."""
    if not SELF_EVAL_ENABLED:
        return False
    level = (state.turn.risk_level or "").lower()
    if level in ("medium", "high"):
        return True
    classification = state.case.classification_output
    if classification and getattr(classification, "confidence", 1.0) < 0.5:
        return True
    text = (state.turn.text or "").lower()
    confusion_cues = ["you didn't answer", "i'm confused", "that doesn't help", "you didn't understand", "not what i asked"]
    if any(c in text for c in confusion_cues):
        return True
    return False


def self_eval(state: AppState):
    """
    Lightweight post-check: did we answer the user's question? At most one follow-up? Safe and appropriate tone?
    If check fails, prepend a short clarification to final_response.
    """
    if not _should_run_self_eval(state):
        return {}
    final = (state.turn.final_response or "").strip()
    user_text = (state.turn.text or "").strip()
    if not final or not user_text:
        return {}
    system_prompt = (
        "## Role\n"
        "You are the quality checker for the counseling assistant's reply. You do not replace the reply; you only suggest a short prepended fix if needed.\n"
        "\n## Background\n"
        "You run only when risk is medium+, or confidence is low, or the user expressed confusion. Your output can prepend one sentence to the existing reply. Reply with OK or FIX: <sentence>.\n"
        "\n## Tasks\n"
        "Check: (1) Did the reply address the user's last message? (2) Is there at most one follow-up question? (3) Is the advice safe and non-extreme? (4) Is the tone appropriate for the user's distress level? Reply with OK or FIX: <one short sentence>.\n"
        "\n## Do\n"
        "Use FIX only when something is clearly wrong or missing; otherwise say OK. Keep FIX to one short sentence.\n"
        "\n## Do not\n"
        "Do not rewrite the whole reply. Do not output long feedback. Do not second-guess minor style.\n"
        "\n## Output format\n"
        "Exactly one line: OK or FIX: <one short sentence to prepend to clarify or fix the reply>"
    )
    user_prompt = f"User said: {user_text}\n\nAssistant replied: {final}"
    raw = (ask_model(system_prompt, user_prompt=user_prompt, history=state.conversation_history) or "").strip()
    if raw.upper().startswith("FIX:"):
        fix = raw[4:].strip()
        if fix:
            state.turn.final_response = fix + " " + (state.turn.final_response or "")
    return {}
