"""Tests for adaptive phase progression logic."""
import pytest
from unittest.mock import patch
from app.agents.phase_manager import (
    _compute_readiness_evidence,
    _decide_transition,
    _apply_transition,
    _detect_soft_signals,
    _milestone_progress,
    phase_manager,
)
from app.models import AppState


class TestMilestoneProgress:
    def test_empty_milestones(self, base_state):
        assert _milestone_progress({}, 1) == 0.0

    def test_full_milestones(self, base_state):
        milestones = {
            "safety_established": True,
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        assert _milestone_progress(milestones, 1) == 1.0

    def test_partial_milestones(self, base_state):
        milestones = {
            "safety_established": True,
            "problem_assessed": True,
        }
        assert _milestone_progress(milestones, 1) == 0.5


class TestSoftSignals:
    def test_reflects_on_own_role(self, base_state):
        base_state.turn.text = "i realize i contributed to this problem"
        signals = _detect_soft_signals(base_state)
        assert "reflects_on_own_role" in signals

    def test_considers_partner_perspective(self, base_state):
        base_state.turn.text = "she feels like i'm not listening"
        signals = _detect_soft_signals(base_state)
        assert "considers_partner_perspective" in signals

    def test_open_to_tools(self, base_state):
        base_state.turn.text = "i'm willing to try that exercise you mentioned"
        signals = _detect_soft_signals(base_state)
        assert "open_to_practical_tools" in signals

    def test_emotional_regulation_by_intensity(self, base_state):
        base_state.turn.text = "things are better this week, we talked calmly"
        base_state.turn.emotional_intensity = 0.1
        signals = _detect_soft_signals(base_state)
        assert "emotional_regulation" in signals


class TestReadinessEvidence:
    def test_low_readiness_early(self, phase1_state):
        phase1_state.therapy.milestones = {}
        score, reason = _compute_readiness_evidence(phase1_state)
        assert score < 0.5
        assert "milestones" in reason

    def test_high_readiness_with_milestones_and_signals(self, phase1_state):
        phase1_state.therapy.milestones = {
            "safety_established": True,
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        phase1_state.turn.text = "i realize i contributed, she feels hurt, i'm willing to try"
        phase1_state.therapy.turns_in_phase = 5
        score, reason = _compute_readiness_evidence(phase1_state)
        assert score >= 0.6

    def test_structural_evidence_from_slots_and_intake(self, phase1_state):
        """Slot completeness and intake_completed contribute to readiness."""
        phase1_state.therapy.milestones = {
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        phase1_state.case.slots_filled = {
            "situation_summary": "schooling disagreement",
            "who_involved": "me and wife",
            "timeframe": "3 months",
            "what_tried": "talked about it",
            "desired_outcome": "find common ground",
        }
        phase1_state.case.intake_completed = True
        phase1_state.therapy.turns_in_phase = 5
        phase1_state.turn.text = "i realize we both want the best for our kid"
        score, reason = _compute_readiness_evidence(phase1_state)
        assert score >= 0.50, f"Expected >=0.50 with 3/4 milestones + full slots + intake, got {score}"
        assert "structural" in reason

    def test_three_of_four_milestones_without_intake_stays_below(self, phase1_state):
        """3/4 milestones without intake or soft signals should not cross gate."""
        phase1_state.therapy.milestones = {
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        phase1_state.case.slots_filled = {}
        phase1_state.case.intake_completed = False
        phase1_state.therapy.turns_in_phase = 4
        phase1_state.turn.text = "ok"
        score, _ = _compute_readiness_evidence(phase1_state)
        assert score < 0.50, f"Should stay below 0.50 without supporting evidence, got {score}"


class TestDecideTransition:
    def test_phase5_never_advances(self, phase5_state):
        decision, reason = _decide_transition(phase5_state, 1.0, "full readiness")
        assert decision == "stay"
        assert "phase_5_terminal" in reason

    def test_below_min_turns_stays(self, phase1_state):
        phase1_state.therapy.turns_in_phase = 1
        decision, reason = _decide_transition(phase1_state, 0.8, "good readiness")
        assert decision == "stay"
        assert "below_min_turns" in reason

    def test_high_readiness_advances(self, phase1_state):
        phase1_state.therapy.turns_in_phase = 4
        decision, reason = _decide_transition(phase1_state, 0.7, "milestone+signals")
        assert decision == "advance"

    def test_hard_cap_triggers_review_not_advance(self, phase1_state):
        """The critical test: hard cap does NOT auto-advance."""
        phase1_state.therapy.turns_in_phase = 6  # min_turns(3) * 2
        decision, reason = _decide_transition(phase1_state, 0.3, "low readiness")
        assert decision == "review_needed"
        assert "hard_cap" in reason

    def test_safety_override_triggers_fallback(self, phase3_state):
        phase3_state.turn.safety_override_triggered = True
        decision, reason = _decide_transition(phase3_state, 0.5, "some readiness")
        assert decision == "temporary_fallback"

    def test_high_distress_triggers_fallback(self, phase4_state):
        phase4_state.turn.emotional_intensity = 0.9
        decision, reason = _decide_transition(phase4_state, 0.5, "ok readiness")
        assert decision == "temporary_fallback"
        assert "high_distress" in reason

    def test_low_readiness_stays(self, phase1_state):
        phase1_state.therapy.turns_in_phase = 5
        decision, reason = _decide_transition(phase1_state, 0.35, "low milestone")
        assert decision == "stay"


class TestApplyTransition:
    def test_advance_increments_phase(self, phase1_state):
        old_phase = phase1_state.therapy.current_phase
        _apply_transition(phase1_state, "advance")
        assert phase1_state.therapy.current_phase == old_phase + 1
        assert phase1_state.therapy.turns_in_phase == 0
        assert phase1_state.therapy.milestones == {}

    def test_temporary_fallback_sets_flag(self, phase3_state):
        _apply_transition(phase3_state, "temporary_fallback")
        assert phase3_state.therapy.temporary_fallback is True
        assert phase3_state.therapy.current_phase == 3  # Phase unchanged

    def test_regress_decrements_phase(self, phase3_state):
        _apply_transition(phase3_state, "regress")
        assert phase3_state.therapy.current_phase == 2

    def test_stay_clears_fallback_when_calm(self, phase3_state):
        phase3_state.therapy.temporary_fallback = True
        phase3_state.turn.emotional_intensity = 0.2
        _apply_transition(phase3_state, "stay")
        assert phase3_state.therapy.temporary_fallback is False

    def test_review_needed_keeps_phase(self, phase1_state):
        _apply_transition(phase1_state, "review_needed")
        assert phase1_state.therapy.current_phase == 1


class TestPhaseManagerNode:
    @patch("app.agents.phase_manager._evaluate_milestones_with_llm")
    def test_phase_manager_sets_decision_metadata(self, mock_eval, phase1_state):
        mock_eval.return_value = {}
        phase_manager(phase1_state)
        assert phase1_state.therapy.phase_transition_decision is not None
        assert phase1_state.therapy.phase_transition_reason is not None
        assert phase1_state.therapy.phase_notes is not None
        assert phase1_state.therapy.total_turns >= 1

    @patch("app.agents.phase_manager._evaluate_milestones_with_llm")
    def test_phase1_stall_scenario_advances_with_mature_evidence(self, mock_eval, phase1_state):
        """Reproduce the two-person Phase 1 schooling stall: 3/4 milestones,
        full slots, intake complete, several turns. Should advance."""
        phase1_state.therapy.turns_in_phase = 5
        phase1_state.therapy.total_turns = 5
        phase1_state.case.slots_filled = {
            "situation_summary": "disagree about child schooling",
            "who_involved": "me and wife",
            "timeframe": "3 months",
            "what_tried": "talked multiple times",
            "desired_outcome": "find a solution together",
        }
        phase1_state.case.intake_completed = True
        mock_eval.return_value = {
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        phase1_state.turn.text = "i realize we both want the best for our child, she feels strongly about this"
        phase_manager(phase1_state)
        assert phase1_state.therapy.phase_transition_decision == "advance"
        assert phase1_state.therapy.current_phase == 2

    @patch("app.agents.phase_manager._evaluate_milestones_with_llm")
    def test_review_needed_state_is_coherent(self, mock_eval, phase1_state):
        """review_needed keeps phase stable and sets clear reason."""
        phase1_state.therapy.turns_in_phase = 5
        phase1_state.therapy.total_turns = 5
        mock_eval.return_value = {}
        phase1_state.turn.text = "ok"
        phase_manager(phase1_state)
        assert phase1_state.therapy.phase_transition_decision == "review_needed"
        assert phase1_state.therapy.current_phase == 1
        assert "hard_cap" in phase1_state.therapy.phase_transition_reason
