"""Psychoeducation agent: explain WHY relationship patterns exist.

When a user asks 'why is this happening?' or 'why are we drifting apart?',
this agent provides insight into the underlying dynamics — attachment styles,
communication patterns, demand-withdraw cycles, etc.
"""
import logging
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState

logger = logging.getLogger(__name__)


def psychoeducation_agent_compute(
    text: str,
    problem_category: str,
    profile_notes: str,
    conversation_history: list,
    detected_horsemen: list | None = None,
    retrieved_info: str = "",
) -> dict:
    """Pure compute: LLM call for psychoeducation insight. Thread-safe."""
    if not text:
        return {}
    knowledge_context = ""
    if retrieved_info.strip():
        knowledge_context = (
            "\n## Retrieved knowledge context\n"
            "Use this only if relevant. Do not quote it verbatim, and do not "
            "mention retrieval or sources in the answer.\n"
            f"{retrieved_info.strip()}\n"
        )
    system_prompt = (
        "## Role\n"
        "You are the psychoeducation specialist for a couples counseling assistant. "
        "You explain WHY relationship patterns happen using accessible language.\n"
        "\n## Background\n"
        "The user is asking why something is happening in their relationship. "
        "Your job is to provide a brief, warm explanation of the underlying dynamic — "
        "NOT to give advice or steps (other agents handle that).\n"
        "\n## Tasks\n"
        "1. Explain the pattern or dynamic in plain language. Use a named pattern only if it adds clarity.\n"
        "2. Explain in 2-3 sentences WHY this happens in relationships — "
        "reference attachment needs, communication styles, or protective behaviours.\n"
        "3. Normalise it: 'This is very common in couples...' or 'Many couples experience this when...'\n"
        "\n## Do\n"
        "- Reference the user's SPECIFIC situation (what they told you)\n"
        "- Use warm, accessible language — no jargon without explanation\n"
        "- Prefer everyday language over clinical labels when the user sounds emotionally raw\n"
        "- Avoid repeating the same pattern name already used in the conversation\n"
        "- Keep it to 2-4 sentences\n"
        "\n## Do not\n"
        "- Do NOT give advice, exercises, or action steps\n"
        "- Do NOT start with 'I hear you' or 'That sounds difficult'\n"
        "- Do NOT be vague — always describe the specific dynamic\n"
        "- Do NOT diagnose or pathologise\n"
        "\n## Output format\n"
        "Return exactly two lines:\n"
        "Insight: <2-4 sentence explanation of WHY this pattern happens, naming the dynamic>\n"
        "Pattern_name: <short name for the pattern, e.g. 'demand-withdraw cycle'>\n"
        "\n## Examples\n"
        "Insight: When one partner keeps missing small bids for attention, the other partner can start "
        "feeling invisible rather than merely disappointed. The pain is not just about the phone or "
        "the silence; it is about the need to matter being left unanswered.\n"
        "Pattern_name: missed connection bids\n"
        "\n"
        f"Problem category: {problem_category or 'Unknown'}\n"
        f"Detected Four Horsemen markers: {', '.join(detected_horsemen or []) or 'none'}\n"
        f"Profile: {profile_notes or 'Not yet collected'}\n"
        f"{knowledge_context}"
    )
    raw = ask_model(
        system_prompt,
        user_prompt=text,
        model=MODEL_NAME,
        history=conversation_history[-6:] if conversation_history else [],
    )
    insight = None
    pattern_name = None
    for line in raw.splitlines():
        lower = line.lower().strip()
        if lower.startswith("insight"):
            insight = line.split(":", 1)[-1].strip()
        elif lower.startswith("pattern_name") or lower.startswith("pattern name"):
            pattern_name = line.split(":", 1)[-1].strip()
    if not insight:
        insight = raw.strip()
    return {
        "psychoeducation_response": insight or "",
        "pattern_name": pattern_name or "",
    }


def psychoeducation_agent(state: AppState):
    """Node wrapper for the psychoeducation agent."""
    if not state.text or not getattr(state.turn, "run_psychoeducation", False):
        return {}
    result = psychoeducation_agent_compute(
        state.text,
        state.problem_category or "",
        state.profile_notes or "",
        state.conversation_history or [],
        state.turn.detected_horsemen or [],
        state.retrieved_info or "",
    )
    if result.get("psychoeducation_response"):
        state.turn.psychoeducation_response = result["psychoeducation_response"]
    return {}
