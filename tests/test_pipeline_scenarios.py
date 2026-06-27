"""End-to-end-ish scenario tests for adaptive workflow.

These test state transitions through individual nodes (not full LLM calls)
by running safety_override -> context_modifier -> phase_manager (mocked LLM)
-> turn_router -> specialist_orchestrator in sequence.
"""
import pytest
from unittest.mock import patch
from app.agents.risk_guard import safety_override
from app.agents.context_modifier import context_modifier
from app.agents.phase_manager import phase_manager
from app.agents.turn_router import turn_router
from app.agents.specialist_orchestrator import specialist_orchestrator
from app.models import AppState


def _run_routing_chain(state):
    """Run the pre-composition routing chain (no LLM calls in phase manager)."""
    safety_override(state)
    context_modifier(state)
    with patch("app.agents.phase_manager._evaluate_milestones_with_llm") as mock_eval:
        mock_eval.return_value = state.therapy.milestones or {}
        phase_manager(state)
    turn_router(state)
    specialist_orchestrator(state)


class TestScenario_EarlyIntakeHighDistress:
    def test_empathy_no_coaching(self, phase1_state):
        phase1_state.turn.text = "i'm so overwhelmed and exhausted, everything is falling apart"
        _run_routing_chain(phase1_state)
        assert phase1_state.turn.turn_mode == "empathy_containment"
        assert phase1_state.turn.run_emotion is True
        assert phase1_state.turn.run_coach is False


class TestScenario_Phase2PracticalRequest:
    def test_mostly_insight_maybe_small_tip(self, base_state):
        """Phase 2 user asks for practical wording."""
        base_state.therapy.current_phase = 2
        base_state.therapy.turns_in_phase = 3
        base_state.therapy.total_turns = 8
        base_state.case.slots_filled = {
            "situation_summary": "wife shuts down",
            "who_involved": "me and wife",
            "timeframe": "3 months",
        }
        base_state.turn.text = "how do i bring up problems without her shutting down"
        _run_routing_chain(base_state)
        # Should get communication coaching or at least psychoeducation
        assert base_state.turn.turn_mode in ("communication_coaching", "psychoeducation", "clarification")
        if base_state.turn.turn_mode != "clarification":
            assert base_state.turn.run_rag is True


class TestScenario_Phase3EmotionallyFlooded:
    def test_containment_first_coaching_delayed(self, phase3_state):
        phase3_state.turn.text = "i'm so overwhelmed and frustrated i can't think straight"
        _run_routing_chain(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"
        assert phase3_state.turn.run_coach is False
        assert phase3_state.turn.run_emotion is True
        assert phase3_state.turn.run_rag is False

    def test_practical_help_runs_coaching_and_rag(self, phase3_state):
        phase3_state.turn.text = "what should i do when our argument starts escalating"
        _run_routing_chain(phase3_state)
        assert phase3_state.turn.turn_mode == "communication_coaching"
        assert phase3_state.turn.run_coach is True
        assert phase3_state.turn.run_rag is True


class TestScenario_Phase4Destabilized:
    def test_temporary_fallback_no_phase_reset(self, phase4_state):
        phase4_state.turn.text = "he brought up the affair again and i'm devastated"
        phase4_state.turn.emotional_intensity = 0.0  # Will be set by router
        _run_routing_chain(phase4_state)
        assert phase4_state.therapy.current_phase == 4  # NOT reset
        assert phase4_state.turn.run_coach is False


class TestScenario_Phase5Maintenance:
    def test_review_and_prevention(self, phase5_state):
        phase5_state.turn.text = "things have been going well, we want to keep improving"
        _run_routing_chain(phase5_state)
        assert phase5_state.turn.turn_mode == "maintenance_review"
        assert phase5_state.turn.run_coach is True or phase5_state.turn.run_growth is True
        assert phase5_state.turn.run_rag is True


class TestScenario_HardCapWithoutReadiness:
    def test_review_needed_not_forced_advance(self, phase1_state):
        """User hits hard cap without enough readiness -> review_needed."""
        phase1_state.therapy.turns_in_phase = 4  # Below hard cap of 6 (min_turns*2)
        phase1_state.therapy.milestones = {}
        _run_routing_chain(phase1_state)
        # Run again to hit cap (turns_in_phase will be incremented to 6 by phase_manager)
        phase1_state.therapy.turns_in_phase = 5
        with patch("app.agents.phase_manager._evaluate_milestones_with_llm") as mock_eval:
            mock_eval.return_value = {}
            phase_manager(phase1_state)
        assert phase1_state.therapy.phase_transition_decision == "review_needed"
        assert phase1_state.therapy.current_phase == 1  # NOT advanced


class TestScenario_NoCoachingLeakagePhase1:
    def test_dissatisfaction_does_not_leak_coaching(self, phase1_state):
        """Phase 1 user expressing dissatisfaction should NOT get coaching."""
        phase1_state.case.slots_filled = {
            "situation_summary": "arguing about money",
            "who_involved": "me and wife",
        }
        phase1_state.case.questions_asked = 4
        phase1_state.turn.text = "anything else? that didn't help"
        _run_routing_chain(phase1_state)
        assert phase1_state.turn.run_coach is False
        assert phase1_state.turn.turn_mode != "communication_coaching"

    def test_resistance_gives_empathy_not_coaching(self, phase1_state):
        """Phase 1 user with enough slots overwhelmed by suggestions gets empathy."""
        phase1_state.case.slots_filled = {
            "situation_summary": "arguing about money",
            "who_involved": "me and wife",
            "timeframe": "1 month",
            "what_tried": "tried talking",
            "desired_outcome": "stop arguing",
        }
        phase1_state.case.questions_asked = 6
        phase1_state.turn.text = "this is too much work, anything easier?"
        _run_routing_chain(phase1_state)
        assert phase1_state.turn.run_coach is False
        assert phase1_state.turn.turn_mode == "empathy_containment"


class TestScenario_PossibleAbuseCoerciveControl:
    def test_no_generic_coaching_for_abuse(self, phase3_state):
        phase3_state.turn.text = "he controls my phone and won't let me see my family"
        _run_routing_chain(phase3_state)
        assert phase3_state.case.context_modifier == "possible_abuse"
        assert phase3_state.turn.run_coach is False
        assert phase3_state.turn.run_rag is False

    def test_safety_flags_set(self, base_state):
        base_state.turn.text = "he controls me and isolates me from friends"
        _run_routing_chain(base_state)
        assert "coercive_control" in base_state.turn.safety_flags


class TestScenario_TrustBreach:
    def test_different_routing_than_ordinary_conflict(self, phase3_state):
        phase3_state.turn.text = "i found out she cheated on me, i'm devastated"
        _run_routing_chain(phase3_state)
        assert phase3_state.case.context_modifier == "repair_after_breach"
        # Should not do ordinary communication coaching
        assert phase3_state.turn.turn_mode == "empathy_containment"


class TestScenario_OneSidedParticipation:
    def test_individual_reflection_routing(self, phase3_state):
        phase3_state.turn.text = "my partner won't come to therapy, i'm doing this alone"
        _run_routing_chain(phase3_state)
        assert phase3_state.case.context_modifier == "one_partner_unavailable"


class TestScenario_InsightActivationOscillation:
    def test_turn_mode_changes_phase_stable(self, phase3_state):
        """User oscillates between insight and activation."""
        # Insight turn
        phase3_state.turn.text = "why does this pattern keep happening with us"
        _run_routing_chain(phase3_state)
        assert phase3_state.turn.turn_mode == "psychoeducation"
        assert phase3_state.turn.run_rag is True
        insight_phase = phase3_state.therapy.current_phase

        # Activation turn
        phase3_state.turn.text = "i'm so frustrated about it now, i can't handle this"
        safety_override(phase3_state)
        context_modifier(phase3_state)
        turn_router(phase3_state)
        specialist_orchestrator(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"
        assert phase3_state.therapy.current_phase == insight_phase


class TestScenario_TwoPersonPhase1Stall:
    """Regression: two-person therapy conversation about schooling stuck in Phase 1."""

    def test_advances_with_mature_conversation(self, base_state):
        """A mature Phase 1 conversation with most milestones, full slots, and
        intake completed should advance to Phase 2."""
        base_state.therapy.current_phase = 1
        base_state.therapy.turns_in_phase = 5
        base_state.therapy.total_turns = 5
        base_state.case.slots_filled = {
            "situation_summary": "disagree about child education approach",
            "who_involved": "me and wife",
            "timeframe": "3 months",
            "what_tried": "discussed several times",
            "desired_outcome": "agree on schooling plan",
        }
        base_state.case.intake_completed = True
        base_state.therapy.milestones = {
            "problem_assessed": True,
            "expectations_set": True,
            "goals_defined": True,
        }
        base_state.turn.text = (
            "i realize we both want what is best, she feels the cost "
            "is too high and i'm willing to try a compromise"
        )
        _run_routing_chain(base_state)
        assert base_state.therapy.phase_transition_decision == "advance"
        assert base_state.therapy.current_phase == 2

    def test_stays_when_truly_early(self, base_state):
        """Minimal context should NOT advance."""
        base_state.therapy.current_phase = 1
        base_state.therapy.turns_in_phase = 2
        base_state.therapy.total_turns = 2
        base_state.therapy.milestones = {}
        base_state.case.slots_filled = {}
        base_state.case.intake_completed = False
        base_state.turn.text = "hello"
        _run_routing_chain(base_state)
        assert base_state.therapy.phase_transition_decision == "stay"
        assert base_state.therapy.current_phase == 1


class TestScenario_IntakeTransitionCopy:
    """Regression: intake copy should not promise phase advancement it does not control."""

    @patch("app.agents.smart_intake.ask_model")
    def test_intake_transition_does_not_claim_phase_jump(self, mock_ask, base_state):
        """When phase_manager kept phase at 1, smart_intake should not tell
        the model to 'transition to the next phase'."""
        from app.agents.smart_intake import smart_intake_agent

        base_state.therapy.current_phase = 1
        base_state.therapy.phase_transition_decision = "stay"
        base_state.case.slots_filled = {
            "situation_summary": "schooling disagreement",
            "who_involved": "me and wife",
            "timeframe": "3 months",
            "what_tried": "tried talking",
            "desired_outcome": "common ground",
        }
        base_state.case.questions_asked = 8
        base_state.turn.text = "yes that works"

        mock_ask.return_value = "I have a good picture of your situation now."
        smart_intake_agent(base_state)

        prompt_sent = mock_ask.call_args[0][0]
        assert "TRANSITION from assessment (Phase 1) to the next phase" not in prompt_sent
        assert "moving to the next phase" not in prompt_sent.lower()

    @patch("app.agents.smart_intake.ask_model")
    def test_intake_transition_acknowledges_advance(self, mock_ask, base_state):
        """When phase_manager actually advanced, intake can acknowledge it."""
        from app.agents.smart_intake import smart_intake_agent

        base_state.therapy.current_phase = 2
        base_state.therapy.phase_transition_decision = "advance"
        base_state.case.slots_filled = {
            "situation_summary": "communication issues",
            "who_involved": "me and partner",
            "timeframe": "2 months",
            "what_tried": "nothing",
            "desired_outcome": "better communication",
        }
        base_state.case.questions_asked = 7
        base_state.turn.text = "ok let's do it"

        mock_ask.return_value = "Great, let's move into deeper work."
        smart_intake_agent(base_state)

        prompt_sent = mock_ask.call_args[0][0]
        assert "progressed to a new phase" in prompt_sent.lower()
