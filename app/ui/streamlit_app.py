"""Streamlit chat front-end for the counseling assistant."""
import logging
import os
import time
import uuid

import streamlit as st

import app  # noqa: F401  — ensure app package is loaded

from app.config import THERAPY_PHASES
from app.logutil import ctx
from app.utils import count_questions

logger = logging.getLogger(__name__)
from app.pipeline import (
    run_pipeline_in_background,
    background_executor,
)


PHASE_LABELS = {
    1: "1 — Assessment",
    2: "2 — Understanding",
    3: "3 — Communication",
    4: "4 — Trust Building",
    5: "5 — Stabilization",
}


def _trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("Streamlit rerun API not available.")


def _reset_session():
    st.session_state.messages = []
    st.session_state.profile_notes = ""
    st.session_state.plan = ""
    st.session_state.formulation_summary = ""
    st.session_state.strengths_summary = ""
    st.session_state.focus_areas = []
    st.session_state.problem_category = ""
    st.session_state.culture = ""
    st.session_state.gender = None
    st.session_state.pending_future = None
    st.session_state.info_future = None
    st.session_state.user_info = ""
    st.session_state.recent_user_messages = []
    st.session_state.pending_follow_up = None
    st.session_state.intake_completed = False
    st.session_state.questions_asked = 0
    st.session_state.active_turn_id = None
    st.session_state.slots_filled = {}
    # Therapy phase state — reset to phase 1
    st.session_state.current_phase = 1
    st.session_state.turns_in_phase = 0
    st.session_state.total_turns = 0
    st.session_state.milestones = {}
    st.session_state.phase_history = []
    st.session_state.therapy_approach = "integrative"
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.therapy_mode = "one_person"
    st.session_state.active_speaker = "A"
    _trigger_rerun()


def _deliver_pending_response():
    pending_future = st.session_state.get("pending_future")
    active_turn_id = st.session_state.get("active_turn_id")
    sid = st.session_state.get("session_id")
    if pending_future:
        logger.info(
            "UI: deliver check %s turn_id_active=%s future_done=%s",
            ctx(None, sid),
            active_turn_id,
            pending_future.done(),
        )
    if not pending_future or not pending_future.done():
        return
    exc = pending_future.exception()
    if exc is not None:
        logger.exception("Pipeline future had exception: %s", exc)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, something went wrong while generating the response: {exc!s}",
        })
        st.session_state.pending_future = None
        return
    try:
        result_state = pending_future.result()
        turn_id = result_state.get("turn_id")
        active_turn_id = st.session_state.get("active_turn_id")
        has_final = bool(result_state.get("final_response"))
        logger.info(
            "UI: deliver pending %s result_turn_id=%s active_turn_id=%s has_final_response=%s result_keys=%s",
            ctx(None, sid),
            turn_id,
            active_turn_id,
            has_final,
            list(result_state.keys()) if isinstance(result_state, dict) else "n/a",
        )
        if not has_final:
            logger.warning("%s Result has no final_response; using fallback message.", ctx(turn_id, sid))
        if turn_id is not None and active_turn_id is not None and turn_id != active_turn_id:
            logger.warning(
                "UI: turn id mismatch, skipping delivery %s result_turn_id=%s active_turn_id=%s",
                ctx(None, sid),
                turn_id,
                active_turn_id,
            )
            st.session_state.pending_future = None
            return
        st.session_state.profile_notes = (
            result_state.get("profile_notes") or st.session_state.profile_notes
        )
        st.session_state.formulation_summary = (
            result_state.get("formulation_summary") or st.session_state.get("formulation_summary", "")
        )
        st.session_state.strengths_summary = (
            result_state.get("strengths_summary") or st.session_state.get("strengths_summary", "")
        )
        if result_state.get("focus_areas") is not None:
            st.session_state.focus_areas = result_state.get("focus_areas") or []
        st.session_state.plan = (
            result_state.get("plan") or st.session_state.plan
        )
        st.session_state.problem_category = (
            result_state.get("problem_category")
            or st.session_state.problem_category
        )
        if result_state.get("slots_filled") is not None:
            st.session_state.slots_filled = result_state.get("slots_filled") or st.session_state.get("slots_filled", {})
        if result_state.get("therapy_mode"):
            st.session_state.therapy_mode = result_state["therapy_mode"]
        assistant_reply = (
            result_state.get("final_response")
            or result_state.get("error")
            or "The assistant couldn't generate a response this time. Try rephrasing or check the terminal for errors."
        )
        st.session_state.pending_follow_up = (
            result_state.get("follow_up_question")
            or st.session_state.get("pending_follow_up")
        )
        has_content = any(
            result_state.get(k)
            for k in [
                "coach_response",
                "growth_response",
                "emotion_response",
                "cultural_note",
                "retrieved_info",
            ]
        )
        if not has_content and not assistant_reply.strip():
            logger.warning(
                "UI: empty delivery, discarding %s result_turn_id=%s",
                ctx(turn_id, sid),
                turn_id,
            )
            st.session_state.pending_future = None
            return

        logger.info(
            "UI: delivered %s reply_len=%s phase=%s",
            ctx(turn_id, sid),
            len(assistant_reply or ""),
            result_state.get("current_phase"),
        )

        # Sync therapy phase state from pipeline
        if result_state.get("current_phase") is not None:
            st.session_state.current_phase = result_state["current_phase"]
        if result_state.get("turns_in_phase") is not None:
            st.session_state.turns_in_phase = result_state["turns_in_phase"]
        if result_state.get("total_turns") is not None:
            st.session_state.total_turns = result_state["total_turns"]
        if result_state.get("milestones") is not None:
            st.session_state.milestones = result_state["milestones"]
        if result_state.get("phase_history") is not None:
            st.session_state.phase_history = result_state["phase_history"]
        # Adaptive routing metadata
        for key in ("turn_mode", "turn_mode_reason", "context_modifier",
                     "safety_override_triggered", "safety_flags",
                     "coaching_eligible", "coaching_eligibility_reason",
                     "readiness_reason", "soft_signals_detected",
                     "phase_transition_decision", "phase_transition_reason",
                     "phase_confidence", "temporary_fallback"):
            if result_state.get(key) is not None:
                st.session_state[key] = result_state[key]

    except Exception as exc:
        assistant_reply = f"Sorry, I ran into an issue pulling the deeper response: {exc}"
        result_state = {}
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    if result_state.get("questions_asked") is not None:
        st.session_state.questions_asked = result_state["questions_asked"]
    else:
        st.session_state.questions_asked += count_questions(assistant_reply)
    if result_state.get("intake_completed"):
        st.session_state.intake_completed = True
    st.session_state.pending_future = None


def _deliver_info_response():
    """Legacy stub — info collection is now handled by smart_intake_agent."""
    info_future = st.session_state.get("info_future")
    if info_future and info_future.done():
        st.session_state.info_future = None


def _render_phase_progress():
    """Render the 5-phase therapy progress bar and current phase info in the sidebar."""
    current = st.session_state.get("current_phase", 1)
    ptd = st.session_state.get("phase_transition_decision")
    is_review = ptd == "review_needed"
    is_fallback = st.session_state.get("temporary_fallback", False)

    st.markdown("### Therapy Progress")

    cols = st.columns(5)
    for i, col in enumerate(cols):
        phase_num = i + 1
        if phase_num < current:
            col.markdown(f"<div style='text-align:center;color:#22c55e;font-weight:bold;'>{phase_num}</div>", unsafe_allow_html=True)
        elif phase_num == current:
            color = "#f59e0b" if is_review else "#3b82f6"
            col.markdown(f"<div style='text-align:center;color:{color};font-weight:bold;font-size:1.2em;'>[ {phase_num} ]</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#9ca3af;'>{phase_num}</div>", unsafe_allow_html=True)

    progress_pct = (current - 1) / 4.0
    if is_review:
        st.progress(progress_pct, text=f"Phase {current} of 5 — review needed")
    elif is_fallback:
        st.progress(progress_pct, text=f"Phase {current} of 5 — temporary fallback")
    else:
        st.progress(progress_pct, text=f"Phase {current} of 5")

    cfg = THERAPY_PHASES.get(current, {})
    st.markdown(f"**{cfg.get('name_en', '')}**")
    st.caption(cfg.get("name_ar", ""))
    st.caption(cfg.get("description", "")[:120] + "...")

    if is_review:
        st.warning("Phase review needed — the session has been in this phase for a while. "
                    "Progress is being evaluated.")
        ptr = st.session_state.get("phase_transition_reason", "")
        if ptr:
            st.caption(f"Reason: {ptr}")
    elif is_fallback:
        st.info("Temporary containment mode — prioritizing emotional safety before "
                "continuing phase work.")

    confidence = st.session_state.get("phase_confidence")
    if confidence is not None and confidence > 0:
        st.caption(f"Readiness: {confidence:.0%}")

    milestones = st.session_state.get("milestones", {})
    expected = cfg.get("milestones", [])
    if expected:
        achieved = sum(1 for m in expected if milestones.get(m))
        st.caption(f"Milestones: {achieved}/{len(expected)}")
        for m in expected:
            icon = "✅" if milestones.get(m) else "⬜"
            label = m.replace("_", " ").title()
            st.caption(f"  {icon} {label}")

    turns = st.session_state.get("turns_in_phase", 0)
    total = st.session_state.get("total_turns", 0)
    st.caption(f"Turns in phase: {turns} | Total turns: {total}")


def run_streamlit_app():
    """Streamlit front-end for chatting with the agent pipeline."""
    st.set_page_config(
        page_title="Couples Counseling Prototype",
        page_icon="💬",
        layout="wide",
    )
    st.title("Mosleh AI: Couples Counseling Assistant")
    st.caption(
        "5-phase couples therapy program with emotion detection, coaching, and cultural guidance."
    )

    defaults = [
        ("messages", []),
        ("profile_notes", ""),
        ("plan", ""),
        ("formulation_summary", ""),
        ("strengths_summary", ""),
        ("focus_areas", []),
        ("problem_category", ""),
        ("gender", None),
        ("culture", ""),
        ("pending_future", None),
        ("info_future", None),
        ("user_info", ""),
        ("recent_user_messages", []),
        ("pending_follow_up", None),
        ("intake_completed", False),
        ("questions_asked", 0),
        ("active_turn_id", None),
        ("slots_filled", {}),
        # Therapy phase defaults
        ("current_phase", 1),
        ("turns_in_phase", 0),
        ("total_turns", 0),
        ("milestones", {}),
        ("phase_history", []),
        ("therapy_approach", "integrative"),
        ("session_id", str(uuid.uuid4())),
        ("therapy_mode", "one_person"),
        ("active_speaker", "A"),
    ]
    for key, default in defaults:
        if key not in st.session_state:
            st.session_state[key] = default

    _deliver_pending_response()
    _deliver_info_response()

    with st.sidebar:
        _render_phase_progress()

        st.markdown("---")
        st.header("Session Settings")
        mode_label = st.selectbox(
            "Therapy mode",
            ["1 person therapy", "2 person therapy"],
            index=0 if st.session_state.therapy_mode == "one_person" else 1,
        )
        st.session_state.therapy_mode = "two_partner" if mode_label.startswith("2") else "one_person"
        if st.session_state.therapy_mode == "two_partner":
            speaker_label = st.radio("Current speaker", ["Partner A", "Partner B"], horizontal=True)
            st.session_state.active_speaker = "A" if speaker_label.endswith("A") else "B"
        else:
            st.session_state.active_speaker = "A"

        gender_choice = st.selectbox(
            "Gender (optional)",
            ["Prefer not to say", "Male", "Female"],
            index=0,
        )
        st.session_state.gender = (
            None if gender_choice == "Prefer not to say" else gender_choice
        )
        st.session_state.culture = st.text_input(
            "Cultural/faith context (optional)",
            value=st.session_state.culture or "",
        )
        st.markdown("**Profile so far**")
        st.write(st.session_state.profile_notes or "Not captured yet.")
        st.markdown("---")
        st.markdown("**Routing info**")
        st.write(f"Plan: {st.session_state.plan or 'N/A'}")
        st.write(
            f"Problem category: {st.session_state.problem_category or 'Unknown'}"
        )
        st.write(f"Therapy mode: {st.session_state.therapy_mode}")
        if st.session_state.formulation_summary:
            st.caption("Formulation")
            st.write(st.session_state.formulation_summary)
        if st.session_state.strengths_summary:
            st.caption("Strengths")
            st.write(st.session_state.strengths_summary)
        if st.session_state.focus_areas:
            st.caption("Focus areas")
            st.write(", ".join(st.session_state.focus_areas))
        # Adaptive routing debug info
        turn_mode = st.session_state.get("turn_mode")
        if turn_mode:
            st.markdown("---")
            st.markdown("**Adaptive Routing**")
            st.write(f"Turn mode: {turn_mode}")
            reason = st.session_state.get("turn_mode_reason", "")
            if reason:
                st.caption(f"Reason: {reason}")
            ctx_mod = st.session_state.get("context_modifier", "")
            if ctx_mod:
                st.write(f"Context: {ctx_mod}")
            coaching = st.session_state.get("coaching_eligible", False)
            st.write(f"Coaching eligible: {'Yes' if coaching else 'No'}")
            cr = st.session_state.get("coaching_eligibility_reason", "")
            if cr:
                st.caption(f"  {cr}")
            safety = st.session_state.get("safety_override_triggered", False)
            if safety:
                flags = st.session_state.get("safety_flags", [])
                st.write(f"Safety override: {', '.join(flags) or 'active'}")
            ptd = st.session_state.get("phase_transition_decision")
            if ptd:
                st.write(f"Phase decision: {ptd}")
            fallback = st.session_state.get("temporary_fallback", False)
            if fallback:
                st.write("Temporary fallback: active")
            signals = st.session_state.get("soft_signals_detected", [])
            if signals:
                st.caption(f"Soft signals: {', '.join(signals)}")
        st.markdown("---")
        st.markdown("**Slots filled**")
        slots = st.session_state.get("slots_filled") or {}
        if slots:
            for k, v in slots.items():
                if v:
                    st.caption(k)
                    st.write(v)
        else:
            st.caption("None yet.")
        st.markdown("---")
        st.markdown("**Captured user info**")
        st.write(st.session_state.user_info or "Not captured yet.")
        if st.button("Reset conversation"):
            _reset_session()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if st.session_state.pending_future:
        st.caption("Gathering deeper guidance from the other agents...")
    if st.session_state.info_future:
        st.caption("Collecting key details about your situation...")

    user_input = st.chat_input("Say hello or describe what's happening...")
    if user_input:
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)
        st.chat_message("user").markdown(user_input)
        st.session_state.recent_user_messages.append(user_input)
        st.session_state.recent_user_messages = (
            st.session_state.recent_user_messages[-5:]
        )

        pending_future = st.session_state.get("pending_future")
        if pending_future and not pending_future.done():
            prev_done = pending_future.done()
            cancelled = pending_future.cancel()
            logger.info(
                "UI: cancel in-flight pipeline %s previous_done=%s cancelled=%s",
                ctx(None, st.session_state.get("session_id")),
                prev_done,
                cancelled,
            )
            st.session_state.pending_future = None
        info_future = st.session_state.get("info_future")
        if info_future and not info_future.done():
            info_future.cancel()
            st.session_state.info_future = None

        turn_id = str(uuid.uuid4())
        st.session_state.active_turn_id = turn_id

        preview = (user_input or "")[:50]
        logger.info(
            "UI: submitting pipeline %s therapy_mode=%s active_speaker=%s questions_asked=%s phase=%s text_len=%s preview=%r",
            ctx(turn_id, st.session_state.session_id),
            st.session_state.get("therapy_mode"),
            st.session_state.get("active_speaker"),
            st.session_state.questions_asked,
            st.session_state.current_phase,
            len(user_input or ""),
            preview,
        )

        pipeline_input = {
            "text": user_input,
            "conversation_history": list(st.session_state.messages),
            "profile_notes": st.session_state.profile_notes or None,
            "user_culture": st.session_state.culture or None,
            "gender": st.session_state.gender,
            "follow_up_question": st.session_state.get("pending_follow_up"),
            "recent_user_messages": st.session_state.recent_user_messages,
            "intake_completed": st.session_state.intake_completed,
            "questions_asked": st.session_state.questions_asked,
            "turn_id": turn_id,
            "slots_filled": st.session_state.get("slots_filled") or {},
            "formulation_summary": st.session_state.get("formulation_summary") or None,
            "strengths_summary": st.session_state.get("strengths_summary") or None,
            "focus_areas": st.session_state.get("focus_areas") or [],
            "therapy_mode": st.session_state.get("therapy_mode") or "one_person",
            "active_speaker": st.session_state.get("active_speaker") or "A",
            "partner_id": st.session_state.get("active_speaker") if st.session_state.get("therapy_mode") == "two_partner" else None,
            # Therapy phase state
            "current_phase": st.session_state.current_phase,
            "session_id": st.session_state.session_id,
            "turns_in_phase": st.session_state.turns_in_phase,
            "total_turns": st.session_state.total_turns,
            "milestones": st.session_state.milestones,
            "phase_history": st.session_state.phase_history,
            "therapy_approach": st.session_state.therapy_approach,
        }
        st.session_state.pending_follow_up = None

        st.session_state.pending_future = background_executor.submit(
            run_pipeline_in_background, pipeline_input
        )

        _deliver_pending_response()

    waiting_on_agents = (
        (
            st.session_state.pending_future
            and not st.session_state.pending_future.done()
        )
        or (
            st.session_state.info_future
            and not st.session_state.info_future.done()
        )
    )
    if waiting_on_agents:
        if os.getenv("MOSLEH_DEBUG_RERUN") or logger.isEnabledFor(logging.DEBUG):
            _pf = st.session_state.pending_future
            _pf_done = _pf.done() if _pf else True
            logger.debug(
                "UI: sleep+rerun %s pending_future_done=%s",
                ctx(st.session_state.get("active_turn_id"), st.session_state.get("session_id")),
                _pf_done,
            )
        time.sleep(0.2)
        _trigger_rerun()
    elif st.session_state.get("pending_future") and st.session_state.pending_future.done():
        _deliver_pending_response()
        _trigger_rerun()


if __name__ == "__main__":
    run_streamlit_app()
