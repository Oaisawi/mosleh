"""Tests for specialist orchestrator — policy-driven agent selection."""
import pytest
from app.agents.specialist_orchestrator import (
    specialist_orchestrator,
    _compute_coaching_readiness,
)
from app.models import AppState


class TestCoachingReadiness:
    def test_abuse_context_blocks_coaching(self, phase3_state):
        phase3_state.case.context_modifier = "possible_abuse"
        eligible, reason = _compute_coaching_readiness(phase3_state)
        assert eligible is False
        assert "unsafe_context" in reason

    def test_high_intensity_blocks_coaching(self, phase3_state):
        phase3_state.turn.emotional_intensity = 0.8
        eligible, reason = _compute_coaching_readiness(phase3_state)
        assert eligible is False
        assert "dysregulated" in reason

    def test_low_slots_blocks_coaching(self, base_state):
        base_state.case.slots_filled = {}
        base_state.turn.emotional_intensity = 0.2
        eligible, reason = _compute_coaching_readiness(base_state)
        assert eligible is False

    def test_good_readiness_allows_coaching(self, phase3_state):
        phase3_state.turn.emotional_intensity = 0.3
        eligible, reason = _compute_coaching_readiness(phase3_state)
        assert eligible is True


class TestSafetyOverrideOrchestration:
    def test_safety_override_emotion_only(self, base_state):
        base_state.turn.safety_override_triggered = True
        base_state.turn.turn_mode = "safety_check"
        specialist_orchestrator(base_state)
        assert base_state.turn.run_emotion is True
        assert base_state.turn.run_coach is False
        assert base_state.turn.run_growth is False


class TestTemporaryFallback:
    def test_fallback_suppresses_coaching(self, phase3_state):
        phase3_state.therapy.temporary_fallback = True
        phase3_state.turn.turn_mode = "empathy_containment"
        specialist_orchestrator(phase3_state)
        assert base_state_run_emotion(phase3_state) is True
        assert phase3_state.turn.run_coach is False
        assert phase3_state.turn.run_growth is False


class TestAbuseContextOrchestration:
    def test_abuse_context_no_coaching(self, phase3_state):
        phase3_state.case.context_modifier = "possible_abuse"
        phase3_state.turn.turn_mode = "empathy_containment"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_coach is False
        assert phase3_state.turn.run_growth is False
        assert phase3_state.turn.run_emotion is True


class TestCoachingTurnMode:
    def test_coaching_mode_with_readiness(self, phase3_state):
        phase3_state.turn.turn_mode = "communication_coaching"
        phase3_state.turn.emotional_intensity = 0.3
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_coach is True
        assert phase3_state.case.coaching_eligible is True

    def test_coaching_mode_without_readiness(self, base_state):
        """Phase 1, low slots -> coaching blocked even if mode requests it."""
        base_state.turn.turn_mode = "communication_coaching"
        base_state.turn.emotional_intensity = 0.2
        base_state.case.slots_filled = {}
        specialist_orchestrator(base_state)
        assert base_state.turn.run_coach is False
        assert base_state.case.coaching_eligible is False


class TestPhase1CoachingBlock:
    def test_phase1_blocks_coaching_even_when_turn_mode_is_coaching(self, base_state):
        """Phase 1 policy blocks coaching unconditionally, regardless of turn_mode."""
        base_state.therapy.current_phase = 1
        base_state.turn.turn_mode = "communication_coaching"
        base_state.turn.emotional_intensity = 0.2
        base_state.case.slots_filled = {
            "situation_summary": "some issue",
            "who_involved": "me and wife",
            "timeframe": "1 month",
            "what_tried": "nothing",
            "desired_outcome": "fix it",
        }
        base_state.profile.profile_notes = "situation: communication issues | involved: wife and me | long note"
        specialist_orchestrator(base_state)
        assert base_state.turn.run_coach is False


class TestPsychoeducationMode:
    def test_psychoeducation_mode_runs_insight(self, phase3_state):
        phase3_state.turn.turn_mode = "psychoeducation"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_psychoeducation is True
        assert phase3_state.turn.run_pattern is True
        assert phase3_state.turn.run_emotion is True
        assert phase3_state.turn.run_rag is True


class TestTheoryTiming:
    def test_trust_repair_high_intensity_runs_emotion_only(self, phase4_state):
        """Trust-repair phase should not force theory while the user is dysregulated."""
        phase4_state.turn.turn_mode = "trust_repair"
        phase4_state.turn.emotional_intensity = 0.85
        phase4_state.case.problem_category = "Emotional Distance"
        specialist_orchestrator(phase4_state)
        assert phase4_state.turn.run_emotion is True
        assert phase4_state.turn.run_coach is False
        assert phase4_state.turn.run_psychoeducation is False
        assert phase4_state.turn.run_pattern is False

    def test_category_overlay_does_not_readd_theory_when_user_is_activated(self, phase3_state):
        """Category modalities cannot reintroduce pattern/psychoeducation past the theory gate."""
        phase3_state.turn.turn_mode = "communication_coaching"
        phase3_state.turn.emotional_intensity = 0.6
        phase3_state.case.problem_category = "Emotional Distance"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_coach is True
        assert phase3_state.turn.run_psychoeducation is False
        assert phase3_state.turn.run_pattern is False


class TestMaintenanceMode:
    def test_maintenance_in_phase5(self, phase5_state):
        phase5_state.turn.turn_mode = "maintenance_review"
        phase5_state.turn.emotional_intensity = 0.2
        specialist_orchestrator(phase5_state)
        assert phase5_state.turn.run_coach is True
        assert phase5_state.turn.run_growth is True
        assert phase5_state.turn.run_rag is True


class TestRagPolicy:
    def test_empathy_containment_skips_rag(self, phase3_state):
        phase3_state.turn.turn_mode = "empathy_containment"
        phase3_state.turn.emotional_intensity = 0.8
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_rag is False

    def test_intake_skips_rag(self, phase3_state):
        phase3_state.turn.turn_mode = "intake_slot_fill"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_rag is False

    def test_clarification_skips_rag(self, phase3_state):
        phase3_state.turn.turn_mode = "clarification"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_rag is False

    def test_coaching_mode_runs_rag_when_ready(self, phase3_state):
        phase3_state.turn.turn_mode = "communication_coaching"
        phase3_state.turn.emotional_intensity = 0.3
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_rag is True
        assert phase3_state.turn.needs_rag is True

    def test_possible_abuse_skips_rag(self, phase3_state):
        phase3_state.case.context_modifier = "possible_abuse"
        phase3_state.turn.turn_mode = "psychoeducation"
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.run_rag is False


def base_state_run_emotion(state):
    return state.turn.run_emotion
