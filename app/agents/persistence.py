"""Persist conversation summaries and therapy phases to SQLite."""
import json
import logging
import sqlite3

from app.models import AppState

logger = logging.getLogger(__name__)

DB_PATH = "conversation_logs.db"


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables(cursor):
    """Create all required tables if they don't exist, and migrate existing ones."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            slots_filled TEXT,
            risk_level TEXT,
            user_intent TEXT,
            assistant_action_type TEXT,
            latency_ms REAL,
            user_text TEXT,
            emotion TEXT,
            assistant_summary TEXT,
            session_id TEXT,
            therapy_phase INTEGER
        )
    """)
    cursor.execute("PRAGMA table_info(conversation_summary)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    migrations = {
        "session_id": "TEXT",
        "therapy_phase": "INTEGER",
        "therapy_mode": "TEXT",
        "active_speaker": "TEXT",
        "partner_id": "TEXT",
        "decision_metadata": "TEXT",
    }
    for col, col_type in migrations.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE conversation_summary ADD COLUMN {col} {col_type}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS therapy_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            current_phase INTEGER DEFAULT 1,
            phase_1_status TEXT DEFAULT 'active',
            phase_2_status TEXT DEFAULT 'pending',
            phase_3_status TEXT DEFAULT 'pending',
            phase_4_status TEXT DEFAULT 'pending',
            phase_5_status TEXT DEFAULT 'pending',
            phase_data TEXT DEFAULT '{}',
            therapy_goals TEXT DEFAULT '[]',
            therapy_approach TEXT DEFAULT 'integrative',
            milestones TEXT DEFAULT '{}',
            total_turns INTEGER DEFAULT 0,
            turns_in_phase INTEGER DEFAULT 0,
            phase_notes TEXT,
            phase_history TEXT DEFAULT '[]',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _build_decision_metadata(state: AppState) -> str:
    """Build a JSON blob of turn-level routing decisions for debugging."""
    return json.dumps({
        "turn_mode": state.turn.turn_mode,
        "turn_mode_reason": state.turn.turn_mode_reason,
        "safety_override_triggered": state.turn.safety_override_triggered,
        "safety_flags": state.turn.safety_flags,
        "context_modifier": state.case.context_modifier,
        "readiness_score": state.case.readiness_score,
        "readiness_reason": state.case.readiness_reason,
        "coaching_eligible": state.case.coaching_eligible,
        "coaching_eligibility_reason": state.case.coaching_eligibility_reason,
        "soft_signals_detected": state.case.soft_signals_detected,
        "milestones_completed": state.case.milestones_completed,
        "phase_transition_decision": state.therapy.phase_transition_decision,
        "phase_transition_reason": state.therapy.phase_transition_reason,
        "phase_confidence": state.therapy.phase_confidence,
        "temporary_fallback": state.therapy.temporary_fallback,
        "risk_level": state.turn.risk_level,
        "risk_type": state.turn.risk_type,
        "risk_action": state.turn.risk_action,
        "allowed_actions": state.turn.allowed_actions,
        "must_ask": state.turn.must_ask,
        "must_refuse": state.turn.must_refuse,
    })


def save_therapy_session(state: AppState):
    """Persist the therapy phase state for the current session."""
    session_id = state.therapy.session_id
    if not session_id:
        return

    conn = _get_connection()
    cursor = conn.cursor()
    _ensure_tables(cursor)

    phase = state.therapy.current_phase
    milestones = json.dumps(state.therapy.milestones or {})
    phase_goals_list = list(state.therapy.phase_goals or [])
    growth_goal = getattr(getattr(state.turn, "growth_output", None), "goal", None)
    if growth_goal and growth_goal not in phase_goals_list:
        phase_goals_list.append(growth_goal)
    phase_goals = json.dumps(phase_goals_list)
    phase_history = json.dumps(state.therapy.phase_history or [])
    phase_notes = state.therapy.phase_notes
    phase_data = json.dumps({
        "therapy_mode": state.case.therapy_mode,
        "formulation_summary": state.case.formulation_summary,
        "strengths_summary": state.case.strengths_summary,
        "focus_areas": state.case.focus_areas,
        "conflict_pattern_assessment": state.case.conflict_pattern_assessment,
        "context_modifier": state.case.context_modifier,
        "phase_confidence": state.therapy.phase_confidence,
        "phase_transition_decision": state.therapy.phase_transition_decision,
        "temporary_fallback": state.therapy.temporary_fallback,
    })

    phase_statuses = {}
    for p in range(1, 6):
        if p < phase:
            phase_statuses[f"phase_{p}_status"] = "completed"
        elif p == phase:
            phase_statuses[f"phase_{p}_status"] = "active"
        else:
            phase_statuses[f"phase_{p}_status"] = "pending"

    cursor.execute("SELECT id FROM therapy_sessions WHERE session_id = ?", (session_id,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            """UPDATE therapy_sessions SET
                current_phase = ?,
                phase_1_status = ?, phase_2_status = ?, phase_3_status = ?,
                phase_4_status = ?, phase_5_status = ?,
                therapy_goals = ?, therapy_approach = ?,
                milestones = ?, total_turns = ?, turns_in_phase = ?,
                phase_notes = ?, phase_history = ?, phase_data = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?""",
            (
                phase,
                phase_statuses["phase_1_status"],
                phase_statuses["phase_2_status"],
                phase_statuses["phase_3_status"],
                phase_statuses["phase_4_status"],
                phase_statuses["phase_5_status"],
                phase_goals,
                state.therapy.therapy_approach or "integrative",
                milestones,
                state.therapy.total_turns,
                state.therapy.turns_in_phase,
                phase_notes,
                phase_history,
                phase_data,
                session_id,
            ),
        )
    else:
        cursor.execute(
            """INSERT INTO therapy_sessions (
                session_id, current_phase,
                phase_1_status, phase_2_status, phase_3_status,
                phase_4_status, phase_5_status,
                therapy_goals, therapy_approach,
                milestones, total_turns, turns_in_phase,
                phase_notes, phase_history, phase_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, phase,
                phase_statuses["phase_1_status"],
                phase_statuses["phase_2_status"],
                phase_statuses["phase_3_status"],
                phase_statuses["phase_4_status"],
                phase_statuses["phase_5_status"],
                phase_goals,
                state.therapy.therapy_approach or "integrative",
                milestones,
                state.therapy.total_turns,
                state.therapy.turns_in_phase,
                phase_notes,
                phase_history,
                phase_data,
            ),
        )

    conn.commit()
    conn.close()
    logger.info(
        "save_therapy_session: session=%s phase=%d turns_in_phase=%d total=%d",
        session_id, phase, state.therapy.turns_in_phase, state.therapy.total_turns,
    )


def load_therapy_session(session_id: str) -> dict:
    """Load therapy phase state from DB. Returns empty dict if not found."""
    if not session_id:
        return {}
    conn = _get_connection()
    cursor = conn.cursor()
    _ensure_tables(cursor)

    cursor.execute("SELECT * FROM therapy_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    phase_data = json.loads(row["phase_data"] or "{}")
    return {
        "current_phase": row["current_phase"],
        "therapy_approach": row["therapy_approach"],
        "milestones": json.loads(row["milestones"] or "{}"),
        "total_turns": row["total_turns"],
        "turns_in_phase": row["turns_in_phase"],
        "phase_goals": json.loads(row["therapy_goals"] or "[]"),
        "phase_notes": row["phase_notes"],
        "phase_history": json.loads(row["phase_history"] or "[]"),
        "therapy_mode": phase_data.get("therapy_mode"),
        "formulation_summary": phase_data.get("formulation_summary"),
        "strengths_summary": phase_data.get("strengths_summary"),
        "focus_areas": phase_data.get("focus_areas") or [],
        "conflict_pattern_assessment": phase_data.get("conflict_pattern_assessment"),
        "context_modifier": phase_data.get("context_modifier"),
        "phase_confidence": phase_data.get("phase_confidence"),
        "phase_transition_decision": phase_data.get("phase_transition_decision"),
        "temporary_fallback": phase_data.get("temporary_fallback", False),
        "session_id": session_id,
    }


def save_conversation_summary(state: AppState):
    """Save one row per turn with decision metadata for debugging."""
    conn = _get_connection()
    cursor = conn.cursor()
    _ensure_tables(cursor)

    turn_id = state.meta.turn_id if state.meta else None
    category = state.case.problem_category if state.case else state.problem_category
    slots_filled = json.dumps(state.case.slots_filled) if state.case and state.case.slots_filled else "{}"
    risk_level = state.turn.risk_level if state.turn else None
    user_intent = state.case.user_intent if state.case else state.user_need
    assistant_action_type = state.turn.dialogue_action if state.turn else None
    latency_ms = state.meta.latency_ms if state.meta else None
    user_text = state.turn.text if state.turn else state.text
    emotion = state.turn.emotion if state.turn else state.emotion
    session_id = state.therapy.session_id if state.therapy else None
    therapy_phase = state.therapy.current_phase if state.therapy else 1
    therapy_mode = state.case.therapy_mode if state.case else "one_person"
    active_speaker = state.turn.active_speaker if state.turn else "A"
    partner_id = state.turn.partner_id if state.turn else None
    decision_metadata = _build_decision_metadata(state)

    if state.problem_category and state.emotion:
        summary_text = f"{state.emotion.capitalize()} about {state.problem_category}"
    else:
        summary_text = (state.text or "")[:100]

    cursor.execute(
        """INSERT INTO conversation_summary (
            turn_id, category, slots_filled, risk_level, user_intent,
            assistant_action_type, latency_ms, user_text, emotion, assistant_summary,
            session_id, therapy_phase, therapy_mode, active_speaker, partner_id,
            decision_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            turn_id,
            category or "Unknown",
            slots_filled,
            risk_level or "none",
            user_intent or "unknown",
            assistant_action_type or "RESPOND_ONLY",
            latency_ms,
            user_text,
            emotion or "Unknown",
            summary_text,
            session_id,
            therapy_phase,
            therapy_mode,
            active_speaker,
            partner_id,
            decision_metadata,
        ),
    )
    conn.commit()
    conn.close()

    save_therapy_session(state)
    return {}
