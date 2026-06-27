"""Pattern/cycle naming agent: identify and reframe recurring relationship cycles.

Helps couples see conflict as a shared cycle ('us vs the pattern')
rather than blaming each other.
"""
import logging
from app.config import MODEL_NAME
from app.llm.providers import ask_model
from app.models import AppState

logger = logging.getLogger(__name__)


def pattern_agent_compute(
    text: str,
    problem_category: str,
    profile_notes: str,
    conversation_history: list,
    detected_horsemen: list | None = None,
    retrieved_info: str = "",
) -> dict:
    """Pure compute: LLM call for pattern/cycle naming. Thread-safe."""
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
        "You are the pattern/cycle specialist for a couples counseling assistant. "
        "You help couples name their recurring cycle and reframe it as 'us vs the pattern'.\n"
        "\n## Background\n"
        "When couples fight repeatedly about the same things, naming the CYCLE "
        "(not blaming either person) helps them step back and work together. "
        "Your job is to name their specific cycle and offer an 'us vs the pattern' reframe.\n"
        "\n## Tasks\n"
        "1. Describe their specific cycle in concrete terms (e.g. 'When you ask for closeness, "
        "she pulls away, which makes you push harder, which makes her withdraw more')\n"
        "2. Name the cycle (e.g. 'pursue-withdraw loop', 'criticism-defensiveness spiral')\n"
        "3. Reframe: position the cycle as the shared enemy, not either partner\n"
        "\n## Do\n"
        "- Use the user's SPECIFIC words and situation\n"
        "- Keep it to 2-3 sentences\n"
        "- Frame it as 'the pattern' or 'the cycle' — externalise it\n"
        "- Be warm and non-blaming\n"
        "\n## Do not\n"
        "- Do NOT give advice or steps (other agents do that)\n"
        "- Do NOT blame either partner\n"
        "- Do NOT start with 'I hear you'\n"
        "- Do NOT be generic — reference their specific situation\n"
        "\n## Output format\n"
        "Return exactly two lines:\n"
        "Cycle: <2-3 sentence description of their specific cycle + reframe>\n"
        "Cycle_name: <short name, e.g. 'silence spiral'>\n"
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
    cycle_text = None
    cycle_name = None
    for line in raw.splitlines():
        lower = line.lower().strip()
        if lower.startswith("cycle:") or lower.startswith("cycle :"):
            cycle_text = line.split(":", 1)[-1].strip()
        elif lower.startswith("cycle_name") or lower.startswith("cycle name"):
            cycle_name = line.split(":", 1)[-1].strip()
    if not cycle_text:
        cycle_text = raw.strip()
    return {
        "pattern_response": cycle_text or "",
        "cycle_name": cycle_name or "",
    }


def pattern_agent(state: AppState):
    """Node wrapper for the pattern/cycle agent."""
    if not state.text or not getattr(state.turn, "run_pattern", False):
        return {}
    result = pattern_agent_compute(
        state.text,
        state.problem_category or "",
        state.profile_notes or "",
        state.conversation_history or [],
        state.turn.detected_horsemen or [],
        state.retrieved_info or "",
    )
    if result.get("pattern_response"):
        state.turn.pattern_response = result["pattern_response"]
    return {}
