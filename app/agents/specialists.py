"""Parallel specialists: run emotion, coach, growth, psychoeducation, pattern agents in parallel."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models import AppState

logger = logging.getLogger(__name__)
from app.agents.emotion import emotion_agent_compute
from app.agents.coach import coach_agent_compute
from app.agents.growth import growth_agent_compute
from app.agents.psychoeducation import psychoeducation_agent_compute
from app.agents.pattern import pattern_agent_compute

# Dedicated executor for parallel specialist LLM calls (max 5 concurrent)
_specialists_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="specialist")


def parallel_specialists(state: AppState) -> dict:
    """
    Run emotion_agent, coach_agent, growth_agent in parallel when run_emotion/run_coach/run_growth
    are True. Merge results into state in the main thread.
    """
    run_emotion = getattr(state.turn, "run_emotion", False)
    run_coach = getattr(state.turn, "run_coach", False)
    run_growth = getattr(state.turn, "run_growth", False)
    run_psychoeducation = getattr(state.turn, "run_psychoeducation", False)
    run_pattern = getattr(state.turn, "run_pattern", False)
    if not (run_emotion or run_coach or run_growth or run_psychoeducation or run_pattern):
        logger.info("parallel_specialists: skipping (all run_* False)")
        return {}

    text = state.turn.text or state.text or ""
    problem_category = state.case.problem_category or state.problem_category or ""
    profile_notes = state.profile.profile_notes or state.profile_notes or ""
    conversation_history = state.conversation_history or []
    readiness = getattr(state.case, "readiness_score", 0.0) or 0.0
    emotion = state.turn.emotion or state.emotion or ""
    slots_filled = state.case.slots_filled or {}
    detected_horsemen = state.turn.detected_horsemen or []
    retrieved_info = state.retrieved_info or ""

    submitted = {}
    if run_emotion and emotion and (emotion or "").lower() != "not enough":
        submitted["emotion"] = _specialists_executor.submit(
            emotion_agent_compute,
            text,
            emotion,
            conversation_history,
        )
    if run_coach:
        submitted["coach"] = _specialists_executor.submit(
            coach_agent_compute,
            text,
            problem_category,
            profile_notes,
            conversation_history,
            readiness,
            response_style=state.turn.response_style or "full_advice",
            slots_filled=slots_filled,
            retrieved_info=retrieved_info,
        )
    if run_growth:
        submitted["growth"] = _specialists_executor.submit(
            growth_agent_compute,
            text,
            problem_category,
            profile_notes,
            conversation_history,
            readiness,
            slots_filled=slots_filled,
            retrieved_info=retrieved_info,
        )
    if run_psychoeducation:
        submitted["psychoeducation"] = _specialists_executor.submit(
            psychoeducation_agent_compute,
            text,
            problem_category,
            profile_notes,
            conversation_history,
            detected_horsemen,
            retrieved_info=retrieved_info,
        )
    if run_pattern:
        submitted["pattern"] = _specialists_executor.submit(
            pattern_agent_compute,
            text,
            problem_category,
            profile_notes,
            conversation_history,
            detected_horsemen,
            retrieved_info=retrieved_info,
        )

    # Build turn-level updates so LangGraph can merge them (in-place mutation may be lost when graph merges state)
    turn_updates = {}
    logger.info("parallel_specialists: submitted %s", list(submitted.keys()))
    future_to_key = {f: k for k, f in submitted.items()}
    for future in as_completed(future_to_key):
        key = future_to_key[future]
        try:
            result = future.result()
        except Exception as e:
            logger.exception("parallel_specialists: %s failed: %s", key, e)
            result = {}
        if not result:
            logger.warning("parallel_specialists: %s returned empty", key)
            continue
        logger.info("parallel_specialists: %s ok", key)
        if key == "emotion":
            if result.get("emotion_output") is not None:
                state.turn.emotion_output = result["emotion_output"]
                turn_updates["emotion_output"] = result["emotion_output"]
            if result.get("emotion_response") is not None:
                state.emotion_response = result["emotion_response"]
                turn_updates["emotion_response"] = result["emotion_response"]
        elif key == "coach":
            if result.get("coach_output") is not None:
                state.turn.coach_output = result["coach_output"]
                turn_updates["coach_output"] = result["coach_output"]
            if result.get("coach_response") is not None:
                state.coach_response = result["coach_response"]
                turn_updates["coach_response"] = result["coach_response"]
        elif key == "growth":
            if result.get("growth_output") is not None:
                state.turn.growth_output = result["growth_output"]
                turn_updates["growth_output"] = result["growth_output"]
            if result.get("growth_response") is not None:
                state.growth_response = result["growth_response"]
                turn_updates["growth_response"] = result["growth_response"]
        elif key == "psychoeducation":
            if result.get("psychoeducation_response"):
                state.turn.psychoeducation_response = result["psychoeducation_response"]
                turn_updates["psychoeducation_response"] = result["psychoeducation_response"]
        elif key == "pattern":
            if result.get("pattern_response"):
                state.turn.pattern_response = result["pattern_response"]
                turn_updates["pattern_response"] = result["pattern_response"]

    if not turn_updates:
        logger.warning("parallel_specialists: no turn_updates collected")
        return {}
    logger.info("parallel_specialists: returning turn_updates keys=%s", list(turn_updates.keys()))
    # Return merged turn so LangGraph merges into state (preserves other turn fields)
    return {"turn": state.turn.model_copy(update=turn_updates)}
