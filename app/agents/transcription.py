"""Voice transcription and text preprocessing."""
import openai

from app.models import AppState


def voice_transcription_agent(state: AppState):
    """Transcribe input audio file to text using OpenAI Whisper."""
    if state.text:
        return {"text": state.text}
    if not state.audio_path:
        raise ValueError("No audio file path provided for transcription.")
    audio_file = open(state.audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    state.text = transcript["text"].strip()
    return {"text": state.text}


def text_preprocessing(state: AppState):
    """Clean up the transcribed text."""
    if not state.text:
        return {}
    cleaned = state.text.strip()
    state.text = cleaned
    return {"text": state.text}
