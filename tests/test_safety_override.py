"""Tests for safety override layer."""
import pytest
from app.agents.risk_guard import safety_override, safety_response
from app.models import AppState


class TestHighRisk:
    def test_suicide_mention_triggers_high_risk(self, base_state):
        base_state.turn.text = "I want to kill myself"
        safety_override(base_state)
        assert base_state.turn.risk_level == "high"
        assert base_state.turn.safety_override_triggered is True
        assert base_state.turn.escalate is True
        assert "self_harm" in base_state.turn.safety_flags

    def test_abuse_mention_triggers_high_risk(self, base_state):
        base_state.turn.text = "he hit me last night"
        safety_override(base_state)
        assert base_state.turn.risk_level == "high"
        assert base_state.turn.safety_override_triggered is True
        assert base_state.turn.must_refuse is not None

    def test_child_abuse_triggers_high_risk(self, base_state):
        base_state.turn.text = "child abuse is happening"
        safety_override(base_state)
        assert base_state.turn.risk_level == "high"
        assert base_state.turn.safety_override_triggered is True


class TestCoerciveControl:
    def test_coercive_control_detected(self, base_state):
        base_state.turn.text = "he controls my phone and won't let me see my friends"
        safety_override(base_state)
        assert base_state.turn.safety_override_triggered is True
        assert "coercive_control" in base_state.turn.safety_flags
        assert base_state.turn.must_refuse is not None
        assert "individual safety" in base_state.turn.must_refuse.lower()

    def test_financial_control(self, base_state):
        base_state.turn.text = "she takes my money and i can't access our account"
        safety_override(base_state)
        assert "coercive_control" in base_state.turn.safety_flags

    def test_isolation_pattern(self, base_state):
        base_state.turn.text = "he cut me off from my family and monitors me"
        safety_override(base_state)
        assert "coercive_control" in base_state.turn.safety_flags
        assert base_state.turn.safety_override_triggered is True


class TestMediumRisk:
    def test_depression_triggers_medium(self, base_state):
        base_state.turn.text = "I feel so depressed and hopeless"
        safety_override(base_state)
        assert base_state.turn.risk_level == "medium"
        assert base_state.turn.must_ask is not None

    def test_severe_escalation_triggers_medium(self, base_state):
        base_state.turn.text = "I'm about to explode, I can't stop myself"
        safety_override(base_state)
        assert base_state.turn.risk_level == "medium"
        assert "severe_escalation" in base_state.turn.safety_flags


class TestPsychiatricFlags:
    def test_psychiatric_red_flag(self, base_state):
        base_state.turn.text = "i've been hearing voices telling me things"
        safety_override(base_state)
        assert "psychiatric_red_flag" in base_state.turn.safety_flags
        assert base_state.turn.must_refuse is not None

    def test_medication_discontinuation(self, base_state):
        base_state.turn.text = "i stopped my meds last week and things are worse"
        safety_override(base_state)
        assert "psychiatric_red_flag" in base_state.turn.safety_flags


class TestNoRisk:
    def test_normal_text_no_flags(self, base_state):
        base_state.turn.text = "we've been arguing about chores lately"
        safety_override(base_state)
        assert base_state.turn.risk_level == "none"
        assert base_state.turn.safety_override_triggered is False
        assert base_state.turn.safety_flags == []

    def test_no_coaching_in_safety_response(self, base_state):
        base_state.turn.text = "I want to kill myself"
        safety_override(base_state)
        safety_response(base_state)
        assert base_state.turn.final_response is not None
        assert base_state.turn.dialogue_action == "RESPOND_ONLY"
