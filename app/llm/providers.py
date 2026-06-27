"""LLM provider wrappers (OpenAI, Gemini). Timeout 90s to avoid indefinite hangs."""
from typing import Dict, List, Optional

import openai
from openai import OpenAI

from app.config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
)

# Timeout in seconds for API calls; prevents pipeline from hanging on a stuck call
LLM_TIMEOUT = 90.0

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)


def ask_openai_chat(
    system_prompt: str,
    user_prompt: str = "",
    model: str = MODEL_NAME,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call OpenAI chat completions."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def ask_gemini_chat(
    system_prompt: str,
    user_prompt: str = "",
    model: str = MODEL_NAME,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call Gemini chat completions using google-generativeai."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini provider selected but google-generativeai is not installed."
        ) from exc
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini provider selected but GEMINI_API_KEY is not set.")

    if not getattr(ask_gemini_chat, "_configured", False):
        genai.configure(api_key=GEMINI_API_KEY)
        ask_gemini_chat._configured = True

    full_prompt_parts: List[str] = []
    if system_prompt:
        full_prompt_parts.append(f"System: {system_prompt}")
    if history:
        hist_lines = [
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history
        ]
        full_prompt_parts.append("\n".join(hist_lines))
    if user_prompt:
        full_prompt_parts.append(f"User: {user_prompt}")
    composite_prompt = "\n\n".join([p for p in full_prompt_parts if p])

    model_name = model or MODEL_NAME
    response = genai.GenerativeModel(model_name).generate_content(composite_prompt)
    return (getattr(response, "text", None) or "").strip()


def ask_model(
    system_prompt: str,
    user_prompt: str = "",
    model: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Unified entry point for chat completions across providers.
    provider: "gemini" or "openai".
    """
    selected_provider = (provider or MODEL_PROVIDER).lower()
    selected_model = model or MODEL_NAME
    if selected_provider == "gemini":
        return ask_gemini_chat(
            system_prompt,
            user_prompt=user_prompt,
            model=selected_model,
            history=history,
        )
    return ask_openai_chat(
        system_prompt,
        user_prompt=user_prompt,
        model=selected_model,
        history=history,
    )
