"""LangGraph pipeline: adaptive safety-first 5-phase therapy graph.

Flow:
  START → ingest → safety_override
    ├── high risk → safety_response → save_summary → END
    └── normal → context_modifier → phase_manager → turn_router
          → specialist_orchestrator
          → [intake_branch or support chain]
          → save_summary → END

The specialist_orchestrator decides run_* flags using phase policy, turn_mode,
safety constraints, and coaching readiness — replacing the old hard phase locks.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

from app.logutil import ctx_from_flat, ctx_from_state
from app.models import AppState
from app.agents.ingest import ingest
from app.agents.risk_guard import safety_override, safety_response
from app.agents.context_modifier import context_modifier
from app.agents.phase_manager import phase_manager
from app.agents.turn_router import turn_router
from app.agents.specialist_orchestrator import specialist_orchestrator
from app.agents import (
    rag_retrieval,
    formulate_response,
    save_conversation_summary,
)
from app.agents.cultural_adapter import cultural_adapter
from app.agents.understanding import combined_understanding
from app.agents.specialists import parallel_specialists
from app.agents.smart_intake import smart_intake_agent
from app.agents.intake_feedback import intake_feedback_evaluation
from app.agents.persistence import load_therapy_session


def _route_after_safety(state: AppState) -> Literal["safety_response", "normal"]:
    """Route: high risk -> safety_response; else -> normal pipeline."""
    level = (state.turn.risk_level or "").lower()
    action = (state.turn.risk_action or "").lower()
    if level == "high" or action == "escalate_to_human":
        decision: Literal["safety_response", "normal"] = "safety_response"
    else:
        decision = "normal"
    logger.info(
        "route_after_safety %s -> %s risk_level=%s risk_action=%s",
        ctx_from_state(state), decision,
        state.turn.risk_level, state.turn.risk_action,
    )
    return decision


def _route_by_turn_mode(state: AppState) -> Literal["intake", "support"]:
    """Route to intake | support based on turn_mode from turn_router."""
    mode = (state.turn.turn_mode or "").lower()
    if mode == "intake_slot_fill":
        decision: Literal["intake", "support"] = "intake"
    else:
        decision = "support"
    logger.info(
        "route_by_turn_mode %s -> %s turn_mode=%s",
        ctx_from_state(state), decision, state.turn.turn_mode,
    )
    return decision


def _route_by_therapy_mode(state: AppState) -> Literal["one_person", "two_partner"]:
    mode = (state.case.therapy_mode or "one_person").lower().strip()
    decision: Literal["one_person", "two_partner"] = "two_partner" if mode == "two_partner" else "one_person"
    logger.info(
        "route_by_therapy_mode %s -> %s therapy_mode=%s",
        ctx_from_state(state), decision, state.case.therapy_mode,
    )
    return decision


def _intake_branch_node(state: AppState):
    """Intake branch: context-aware smart intake agent."""
    smart_intake_agent(state)
    return {}


def _intake_branch_two_partner_node(state: AppState):
    """Two-partner intake branch."""
    smart_intake_agent(state)
    return {}


def _route_after_intake(state: AppState) -> Literal["intake_feedback", "save_summary"]:
    """After intake: if intake is complete, transition through feedback to support."""
    intake_completed = state.case.intake_completed
    turn_type = (state.turn.turn_type or "").lower()

    if intake_completed or turn_type == "intake_to_support":
        decision: Literal["intake_feedback", "save_summary"] = "intake_feedback"
    else:
        decision = "save_summary"
    logger.info(
        "route_after_intake %s -> %s intake_completed=%s turn_type=%s",
        ctx_from_state(state), decision, intake_completed, state.turn.turn_type,
    )
    return decision


def _intake_feedback_node(state: AppState):
    """Post-intake therapist feedback node before specialist support path."""
    intake_feedback_evaluation(state)
    return {}


def _support_entry_two_partner_node(state: AppState):
    """Two-partner support branch marker (orchestrator already ran)."""
    if state.case.plan:
        state.case.plan = f"[two_partner] {state.case.plan}"
    return {}


def _build_graph():
    graph = StateGraph(AppState)

    # Core pipeline nodes
    graph.add_node("ingest", ingest)
    graph.add_node("safety_override", safety_override)
    graph.add_node("safety_response", safety_response)
    graph.add_node("context_modifier", context_modifier)
    graph.add_node("phase_manager", phase_manager)
    graph.add_node("turn_router", turn_router)
    graph.add_node("specialist_orchestrator", specialist_orchestrator)

    # Branch / dispatch nodes
    graph.add_node("mode_dispatch", lambda s: {})
    graph.add_node("one_person_route", lambda s: {})
    graph.add_node("two_partner_route", lambda s: {})
    graph.add_node("dispatch", lambda s: {})
    graph.add_node("dispatch_two_partner", lambda s: {})
    graph.add_node("intake_branch", _intake_branch_node)
    graph.add_node("intake_branch_two_partner", _intake_branch_two_partner_node)
    graph.add_node("intake_feedback", _intake_feedback_node)
    graph.add_node("intake_feedback_two_partner", _intake_feedback_node)
    graph.add_node("support_entry_two_partner", _support_entry_two_partner_node)

    # Support chain nodes
    graph.add_node("rag_retrieval", rag_retrieval)
    graph.add_node("combined_understanding", combined_understanding)
    graph.add_node("parallel_specialists", parallel_specialists)
    graph.add_node("cultural_adapter", cultural_adapter)
    graph.add_node("formulate_response", formulate_response)
    graph.add_node("save_summary", save_conversation_summary)

    # === Edges ===
    # START → ingest → safety_override
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "safety_override")

    # safety_override → safety_response (high risk) OR normal pipeline
    graph.add_conditional_edges("safety_override", _route_after_safety, {
        "safety_response": "safety_response",
        "normal": "context_modifier",
    })
    graph.add_edge("safety_response", "save_summary")

    # context_modifier → phase_manager → turn_router → specialist_orchestrator
    graph.add_edge("context_modifier", "phase_manager")
    graph.add_edge("phase_manager", "turn_router")
    graph.add_edge("turn_router", "specialist_orchestrator")

    # specialist_orchestrator → mode dispatch (one_person vs two_partner)
    graph.add_edge("specialist_orchestrator", "mode_dispatch")
    graph.add_conditional_edges("mode_dispatch", _route_by_therapy_mode, {
        "one_person": "one_person_route",
        "two_partner": "two_partner_route",
    })
    graph.add_edge("one_person_route", "dispatch")
    graph.add_edge("two_partner_route", "dispatch_two_partner")

    # dispatch → intake or support
    graph.add_conditional_edges("dispatch", _route_by_turn_mode, {
        "intake": "intake_branch",
        "support": "rag_retrieval",
    })
    graph.add_conditional_edges("dispatch_two_partner", _route_by_turn_mode, {
        "intake": "intake_branch_two_partner",
        "support": "support_entry_two_partner",
    })

    # Intake branch → conditionally route to support or save
    graph.add_conditional_edges("intake_branch", _route_after_intake, {
        "intake_feedback": "intake_feedback",
        "save_summary": "save_summary",
    })
    graph.add_conditional_edges("intake_branch_two_partner", _route_after_intake, {
        "intake_feedback": "intake_feedback_two_partner",
        "save_summary": "save_summary",
    })
    graph.add_edge("intake_feedback", "rag_retrieval")
    graph.add_edge("intake_feedback_two_partner", "support_entry_two_partner")

    # Two-partner support -> rag
    graph.add_edge("support_entry_two_partner", "rag_retrieval")

    # Support chain: rag → understanding → specialists → cultural → compose → save
    graph.add_edge("rag_retrieval", "combined_understanding")
    graph.add_edge("combined_understanding", "parallel_specialists")
    graph.add_edge("parallel_specialists", "cultural_adapter")
    graph.add_edge("cultural_adapter", "formulate_response")
    graph.add_edge("formulate_response", "save_summary")

    graph.add_edge("save_summary", END)

    return graph.compile()


pipeline = _build_graph()
background_executor = ThreadPoolExecutor(max_workers=2)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get key from dict or attribute from object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _flatten_nested_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested LangGraph state (turn, case, profile, meta, therapy) for UI."""
    turn = raw.get("turn") or {}
    case = raw.get("case") or {}
    profile = raw.get("profile") or {}
    meta = raw.get("meta") or {}
    therapy = raw.get("therapy") or {}
    return {
        "text": _get(turn, "text"),
        "audio_path": _get(turn, "audio_path"),
        "recent_user_messages": _get(turn, "recent_user_messages") or raw.get("conversation_history"),
        "emotion": _get(turn, "emotion"),
        "sentiment": _get(turn, "sentiment"),
        "problem_category": _get(case, "problem_category"),
        "plan": _get(case, "plan"),
        "therapy_mode": _get(case, "therapy_mode", "one_person"),
        "formulation_summary": _get(case, "formulation_summary"),
        "strengths_summary": _get(case, "strengths_summary"),
        "focus_areas": _get(case, "focus_areas", []),
        "conflict_pattern_assessment": _get(case, "conflict_pattern_assessment"),
        "emotion_response": _get(turn, "emotion_response"),
        "coach_response": _get(turn, "coach_response"),
        "growth_response": _get(turn, "growth_response"),
        "retrieved_info": _get(turn, "retrieved_info"),
        "cultural_note": _get(turn, "cultural_note"),
        "final_response": _get(turn, "final_response"),
        "profile_notes": _get(profile, "profile_notes"),
        "follow_up_question": _get(turn, "follow_up_question"),
        "active_speaker": _get(turn, "active_speaker", "A"),
        "partner_id": _get(turn, "partner_id"),
        "user_culture": _get(profile, "culture"),
        "user_need": _get(case, "user_intent"),
        "turn_id": _get(meta, "turn_id"),
        "need_emotion": _get(turn, "run_emotion"),
        "need_coach": _get(turn, "run_coach"),
        "need_growth": _get(turn, "run_growth"),
        "needs_rag": _get(turn, "run_rag"),
        "need_psychoeducation": _get(turn, "run_psychoeducation"),
        "need_pattern": _get(turn, "run_pattern"),
        "psychoeducation_response": _get(turn, "psychoeducation_response"),
        "pattern_response": _get(turn, "pattern_response"),
        "detected_horsemen": _get(turn, "detected_horsemen", []),
        "intake_completed": _get(case, "intake_completed"),
        "questions_asked": _get(case, "questions_asked"),
        "slots_filled": _get(case, "slots_filled"),
        "readiness_score": _get(case, "readiness_score"),
        "context_modifier": _get(case, "context_modifier"),
        "readiness_reason": _get(case, "readiness_reason"),
        "soft_signals_detected": _get(case, "soft_signals_detected", []),
        "milestones_completed": _get(case, "milestones_completed", []),
        "coaching_eligible": _get(case, "coaching_eligible", False),
        "coaching_eligibility_reason": _get(case, "coaching_eligibility_reason"),
        "turn_mode": _get(turn, "turn_mode"),
        "turn_mode_reason": _get(turn, "turn_mode_reason"),
        "safety_override_triggered": _get(turn, "safety_override_triggered", False),
        "safety_flags": _get(turn, "safety_flags", []),
        # Therapy phase state
        "current_phase": _get(therapy, "current_phase", 1),
        "session_id": _get(therapy, "session_id"),
        "turns_in_phase": _get(therapy, "turns_in_phase", 0),
        "total_turns": _get(therapy, "total_turns", 0),
        "phase_goals": _get(therapy, "phase_goals", []),
        "therapy_approach": _get(therapy, "therapy_approach", "integrative"),
        "milestones": _get(therapy, "milestones", {}),
        "phase_notes": _get(therapy, "phase_notes"),
        "phase_history": _get(therapy, "phase_history", []),
        "phase_confidence": _get(therapy, "phase_confidence", 1.0),
        "phase_transition_decision": _get(therapy, "phase_transition_decision"),
        "phase_transition_reason": _get(therapy, "phase_transition_reason"),
        "temporary_fallback": _get(therapy, "temporary_fallback", False),
    }


def normalize_state(raw_state) -> Dict[str, Any]:
    """Convert LangGraph output into a plain dict. Flatten nested state for UI."""
    if hasattr(raw_state, "model_dump"):
        return raw_state.model_dump()
    if hasattr(raw_state, "dict"):
        return raw_state.dict()
    if isinstance(raw_state, dict):
        if "turn" in raw_state:
            return _flatten_nested_state(raw_state)
        return raw_state
    return dict(raw_state)


def run_pipeline_in_background(pipeline_input: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the pipeline and normalize the state (for threaded execution)."""
    t0 = time.perf_counter()
    turn_id = pipeline_input.get("turn_id") if isinstance(pipeline_input, dict) else None
    therapy_mode = (pipeline_input or {}).get("therapy_mode") if isinstance(pipeline_input, dict) else None
    current_phase = (pipeline_input or {}).get("current_phase") if isinstance(pipeline_input, dict) else None
    logger.info(
        "Pipeline started %s text_len=%s therapy_mode=%s current_phase=%s",
        ctx_from_flat(pipeline_input if isinstance(pipeline_input, dict) else None),
        len((pipeline_input or {}).get("text") or ""),
        therapy_mode,
        current_phase,
    )
    try:
        if isinstance(pipeline_input, dict):
            keys_before = set(pipeline_input.keys())
            session_id = pipeline_input.get("session_id")
            if session_id:
                persisted = load_therapy_session(session_id)
                if persisted:
                    for k, v in persisted.items():
                        pipeline_input.setdefault(k, v)
                    merged = [k for k in persisted if k not in keys_before]
                    if merged:
                        logger.info(
                            "Pipeline load_therapy_session %s merged_missing_keys=%s",
                            ctx_from_flat(pipeline_input),
                            sorted(merged),
                        )
        initial = AppState.from_flat_dict(pipeline_input) if isinstance(pipeline_input, dict) else pipeline_input
        if not isinstance(initial, AppState):
            initial = AppState.from_flat_dict(initial)
        raw_state = pipeline.invoke(initial)
        result = normalize_state(raw_state)
        if isinstance(pipeline_input, dict) and pipeline_input.get("turn_id") and "turn_id" not in result:
            result["turn_id"] = pipeline_input["turn_id"]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        has_final = bool(result.get("final_response"))
        has_tid = isinstance(result, dict) and bool(result.get("turn_id"))
        logger.info(
            "Pipeline completed %s has_final_response=%s result_has_turn_id=%s elapsed_ms=%.1f result_keys=%s",
            ctx_from_flat(pipeline_input if isinstance(pipeline_input, dict) else None),
            has_final,
            has_tid,
            elapsed_ms,
            list(result.keys()) if isinstance(result, dict) else "n/a",
        )
        if not has_final:
            logger.warning(
                "Pipeline returned no final_response; result may be incomplete. %s",
                ctx_from_flat(pipeline_input if isinstance(pipeline_input, dict) else None),
            )
        return result
    except Exception as e:
        logger.exception(
            "Pipeline failed %s: %s",
            ctx_from_flat(pipeline_input if isinstance(pipeline_input, dict) else None),
            e,
        )
        raise
