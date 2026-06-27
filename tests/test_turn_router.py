"""Tests for turn router — adaptive turn-mode selection."""
import pytest
from app.agents.turn_router import turn_router
from app.models import AppState


class TestDistressContainment:
    def test_high_distress_routes_to_containment(self, phase3_state):
        """Phase 3 user who is emotionally flooded -> containment, not coaching."""
        phase3_state.turn.text = "i'm so overwhelmed and exhausted, i can't take this anymore"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"
        assert phase3_state.turn.response_style == "empathy_only"

    def test_phase1_distress_also_contained(self, phase1_state):
        """Early intake high distress -> containment."""
        phase1_state.turn.text = "i feel so hopeless and exhausted"
        turn_router(phase1_state)
        assert phase1_state.turn.turn_mode == "empathy_containment"

    def test_distress_with_help_request(self, phase3_state):
        """Distressed but asking for help -> still gets help."""
        phase3_state.turn.text = "i'm upset but what should i do about the argument"
        turn_router(phase3_state)
        # Should not be pure containment since they want help
        assert phase3_state.turn.turn_mode in ("communication_coaching", "empathy_containment")


class TestShameDignityContainment:
    def test_shame_dignity_routes_to_containment_before_phase_default(self, phase4_state):
        """Worth/shame pain should get attunement instead of phase-4 trust repair."""
        phase4_state.turn.text = "i feel like i'm not worth that effort anymore"
        turn_router(phase4_state)
        assert phase4_state.turn.turn_mode == "empathy_containment"
        assert phase4_state.turn.response_style == "empathy_only"
        assert phase4_state.turn.emotional_intensity >= 0.8
        assert "shame_dignity" in phase4_state.turn.turn_mode_reason

    def test_shame_dignity_with_direct_help_request_can_still_get_tools(self, phase3_state):
        """A clear help request can still route to coaching rather than pure containment."""
        phase3_state.turn.text = "i feel small, but what should i do tonight?"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "communication_coaching"


class TestPhase2DirectRequest:
    def test_understanding_cues_in_phase2(self, base_state):
        """Phase 2 user asking 'why' -> psychoeducation mode."""
        base_state.therapy.current_phase = 2
        base_state.turn.text = "why does she always shut down when i bring up problems"
        turn_router(base_state)
        assert base_state.turn.turn_mode == "psychoeducation"


class TestPhaseDefaults:
    def test_phase1_default_is_intake(self, phase1_state):
        """Phase 1 with generic text -> intake slot fill."""
        phase1_state.turn.text = "we've been having some issues lately"
        turn_router(phase1_state)
        assert phase1_state.turn.turn_mode == "intake_slot_fill"

    def test_phase5_default_is_maintenance(self, phase5_state):
        """Phase 5 generic -> maintenance review."""
        phase5_state.turn.text = "things have been going well"
        turn_router(phase5_state)
        assert phase5_state.turn.turn_mode == "maintenance_review"


class TestSafetyOverrideActive:
    def test_safety_override_routes_to_safety_check(self, base_state):
        """When safety override is already triggered upstream."""
        base_state.turn.safety_override_triggered = True
        base_state.turn.text = "anything"
        turn_router(base_state)
        assert base_state.turn.turn_mode == "safety_check"

    def test_risk_keywords_route_to_safety(self, base_state):
        base_state.turn.text = "i want to end my life"
        turn_router(base_state)
        assert base_state.turn.turn_mode == "safety_check"


class TestAbuseContext:
    def test_possible_abuse_context_routes_containment(self, phase3_state):
        """Abuse context modifier -> containment even in later phase."""
        phase3_state.case.context_modifier = "possible_abuse"
        phase3_state.turn.text = "i'm scared of what happens next"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"


class TestPhase4Destabilized:
    def test_temporary_fallback_routes_containment(self, phase4_state):
        """Phase 4 user in temporary fallback -> containment without phase reset."""
        phase4_state.therapy.temporary_fallback = True
        phase4_state.turn.text = "everything just came crashing down when he mentioned the affair"
        turn_router(phase4_state)
        assert phase4_state.turn.turn_mode == "empathy_containment"
        assert phase4_state.therapy.current_phase == 4  # Phase not reset


class TestOscillation:
    def test_insight_then_activation_stable_phase(self, phase3_state):
        """Turn mode changes adaptively while phase remains stable."""
        # First: understanding turn
        phase3_state.turn.text = "why does this pattern keep happening"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "psychoeducation"
        old_phase = phase3_state.therapy.current_phase

        # Second: activated turn
        phase3_state.turn.text = "i'm so frustrated i can't deal with this"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"
        assert phase3_state.therapy.current_phase == old_phase


class TestResistancePath:
    def test_resistance_routes_to_containment(self, phase3_state):
        """User overwhelmed by advice -> low-burden empathy, not more coaching."""
        phase3_state.turn.text = "this seems like a lot of work, anything simpler?"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "empathy_containment"
        assert "resistance" in phase3_state.turn.turn_mode_reason

    def test_resistance_in_phase1_with_low_slots(self, phase1_state):
        """Resistance in phase 1 with low completeness routes to intake first."""
        phase1_state.turn.text = "that's too complicated for us right now"
        turn_router(phase1_state)
        assert phase1_state.turn.turn_mode == "intake_slot_fill"

    def test_resistance_in_phase1_with_filled_slots(self, phase1_state):
        """Resistance in phase 1 with filled slots routes to containment."""
        phase1_state.case.slots_filled = {
            "situation_summary": "arguments",
            "who_involved": "us",
            "timeframe": "2 weeks",
            "what_tried": "talking",
            "desired_outcome": "peace",
        }
        phase1_state.case.questions_asked = 6
        phase1_state.turn.text = "that's too complicated for us right now"
        turn_router(phase1_state)
        assert phase1_state.turn.turn_mode == "empathy_containment"


class TestDissatisfactionInBlockedPhase:
    def test_dissatisfaction_no_coaching_in_phase1(self, phase1_state):
        """Phase 1 blocks coaching, so dissatisfaction routes to psychoeducation."""
        phase1_state.case.slots_filled = {
            "situation_summary": "arguments",
            "who_involved": "us",
            "timeframe": "2 weeks",
        }
        phase1_state.case.questions_asked = 3
        phase1_state.turn.text = "anything else you can suggest? didn't work"
        turn_router(phase1_state)
        assert phase1_state.turn.turn_mode != "communication_coaching"

    def test_dissatisfaction_allows_coaching_in_phase3(self, phase3_state):
        """Phase 3 allows coaching, so dissatisfaction gets coaching."""
        phase3_state.turn.text = "anything else? that didn't work for us"
        turn_router(phase3_state)
        assert phase3_state.turn.turn_mode == "communication_coaching"


class TestHorsemenDetection:
    def test_four_horsemen_detected(self, base_state):
        base_state.turn.text = "you always do this, you never listen, it's your fault"
        base_state.therapy.current_phase = 3
        turn_router(base_state)
        assert "criticism" in base_state.turn.detected_horsemen
