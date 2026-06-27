"""Utility functions for intake, profile, and formatting."""
from typing import Dict, Optional

from app.config import GREETINGS, INTAKE_QUESTIONS


def is_light_intake_input(text: str) -> bool:
    normalized = (text or "").lower().strip()
    if not normalized:
        return True
    return len(normalized) < 20 or any(word in normalized for word in GREETINGS)


def count_questions(text: str) -> int:
    if not text:
        return 0
    return 1 if "?" in text else 0


def pick_intake_question(questions_asked: int) -> str:
    if not INTAKE_QUESTIONS:
        return "Can you tell me a bit more about what's going on?"
    index = min(max(questions_asked, 0), len(INTAKE_QUESTIONS) - 1)
    return INTAKE_QUESTIONS[index]


def profile_is_collected(profile_notes: Optional[str]) -> bool:
    if not profile_notes:
        return False
    normalized = profile_notes.strip().lower()
    if not normalized:
        return False
    if normalized in {"no profile yet.", "not captured yet."}:
        return False
    return True


def format_known_info(known_info: Optional[Dict[str, object]]) -> str:
    if not known_info:
        return "None"
    parts = []
    for key, value in known_info.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            rendered = ", ".join(str(v) for v in value if v)
        else:
            rendered = str(value).strip()
        if not rendered:
            continue
        parts.append(f"- {key}: {rendered}")
    return "\n".join(parts) if parts else "None"
