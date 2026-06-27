"""Tests that retrieved context is passed only to knowledge/action specialists."""
from app.agents import specialists


def test_parallel_specialists_passes_retrieved_info_to_grounded_specialists(
    phase3_state,
    monkeypatch,
):
    received = {}

    def fake_emotion(*args, **kwargs):
        received["emotion_kwargs"] = kwargs
        return {"emotion_response": "emotion response"}

    def fake_coach(*args, **kwargs):
        received["coach_retrieved"] = kwargs.get("retrieved_info")
        return {"coach_response": "coach response"}

    def fake_growth(*args, **kwargs):
        received["growth_retrieved"] = kwargs.get("retrieved_info")
        return {"growth_response": "growth response"}

    def fake_psychoeducation(*args, **kwargs):
        received["psychoeducation_retrieved"] = kwargs.get("retrieved_info")
        return {"psychoeducation_response": "insight response"}

    def fake_pattern(*args, **kwargs):
        received["pattern_retrieved"] = kwargs.get("retrieved_info")
        return {"pattern_response": "pattern response"}

    monkeypatch.setattr(specialists, "emotion_agent_compute", fake_emotion)
    monkeypatch.setattr(specialists, "coach_agent_compute", fake_coach)
    monkeypatch.setattr(specialists, "growth_agent_compute", fake_growth)
    monkeypatch.setattr(specialists, "psychoeducation_agent_compute", fake_psychoeducation)
    monkeypatch.setattr(specialists, "pattern_agent_compute", fake_pattern)

    phase3_state.turn.run_emotion = True
    phase3_state.turn.run_coach = True
    phase3_state.turn.run_growth = True
    phase3_state.turn.run_psychoeducation = True
    phase3_state.turn.run_pattern = True
    phase3_state.turn.emotion = "stressed"
    phase3_state.case.problem_category = "Communication"
    phase3_state.retrieved_info = "grounded qdrant context"

    specialists.parallel_specialists(phase3_state)

    assert received["emotion_kwargs"] == {}
    assert received["coach_retrieved"] == "grounded qdrant context"
    assert received["growth_retrieved"] == "grounded qdrant context"
    assert received["psychoeducation_retrieved"] == "grounded qdrant context"
    assert received["pattern_retrieved"] == "grounded qdrant context"
