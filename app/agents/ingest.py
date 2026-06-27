"""Ingest: transcription + preprocessing. Single node at graph start."""
import logging

from app.agents.transcription import text_preprocessing, voice_transcription_agent
from app.logutil import ctx_from_state
from app.models import AppState

logger = logging.getLogger(__name__)


def ingest(state: AppState):
    """Run voice_transcription (if audio) then text_preprocessing. Output: state.turn.text."""
    audio = bool(getattr(state, "audio_path", None) or getattr(state.turn, "audio_path", None))
    text_len = len((state.turn.text or state.text or "") or "")
    logger.info(
        "ingest %s text_len=%s has_audio_path=%s",
        ctx_from_state(state),
        text_len,
        audio,
    )
    # Ensure turn has input from top-level (UI sends flat dict that becomes AppState)
    if state.text is not None:
        state.turn.text = state.text
    if state.audio_path is not None:
        state.turn.audio_path = state.audio_path
    if state.recent_user_messages:
        state.turn.recent_user_messages = state.recent_user_messages
    voice_transcription_agent(state)
    text_preprocessing(state)
    state.turn.text = state.text
    return {}
