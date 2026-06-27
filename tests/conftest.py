"""Shared fixtures for the adaptive workflow tests."""
import pytest
from app.models import AppState


@pytest.fixture
def base_state():
    """Minimal AppState for unit tests."""
    return AppState.from_flat_dict({
        "text": "",
        "session_id": "test-session",
        "conversation_history": [],
    })


@pytest.fixture
def phase1_state(base_state):
    """State in Phase 1 with some history."""
    base_state.therapy.current_phase = 1
    base_state.therapy.turns_in_phase = 2
    base_state.therapy.total_turns = 2
    base_state.turn.text = "my wife and I have communication problems"
    base_state.case.slots_filled = {"situation_summary": "communication problems"}
    return base_state


@pytest.fixture
def phase3_state(base_state):
    """State in Phase 3 with reasonable readiness."""
    base_state.therapy.current_phase = 3
    base_state.therapy.turns_in_phase = 3
    base_state.therapy.total_turns = 12
    base_state.case.slots_filled = {
        "situation_summary": "communication issues",
        "who_involved": "me and wife",
        "timeframe": "6 months",
        "what_tried": "nothing yet",
        "desired_outcome": "better communication",
    }
    base_state.profile.profile_notes = "situation_summary: communication issues | who_involved: me and wife | timeframe: 6 months"
    base_state.turn.text = ""
    return base_state


@pytest.fixture
def phase4_state(phase3_state):
    """State in Phase 4."""
    phase3_state.therapy.current_phase = 4
    phase3_state.therapy.turns_in_phase = 2
    phase3_state.therapy.total_turns = 18
    return phase3_state


@pytest.fixture
def phase5_state(phase3_state):
    """State in Phase 5."""
    phase3_state.therapy.current_phase = 5
    phase3_state.therapy.turns_in_phase = 1
    phase3_state.therapy.total_turns = 22
    return phase3_state
