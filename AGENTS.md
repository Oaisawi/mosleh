# New Agent Project Context: Mosleh AI (Capstone2)

## 1) What this project is
Mosleh AI is a single-service Streamlit + LangGraph couples-counseling assistant.
It runs as one Python app with:
- no external DB server (SQLite file: `conversation_logs.db`)
- optional Qdrant vector DB for grounded RAG, with in-process FAISS fallback
- OpenAI (default) or Gemini via env switch
- a 5-phase therapy workflow with adaptive safety-first routing

Primary runtime entry points:
- **`python -m streamlit run app/ui/streamlit_app.py`** — recommended (from repo root).
- **`run_app.py`** — imports `run_streamlit_app()` from `streamlit_app.py`; use **`streamlit run run_app.py`** if you want this script as the CLI target (avoid `python run_app.py` unless you know your Streamlit version supports it).
- **`app/ui/streamlit_app.py`** — chat UI + async pipeline calls (`run_streamlit_app`).
- **`app/pipeline.py`** — LangGraph DAG and routing logic.

## 2) How to run
Install deps:
```bash
pip install -r requirements.txt
```

Run app (recommended):
```bash
python -m streamlit run app/ui/streamlit_app.py
```

Optional headless:
```bash
streamlit run app/ui/streamlit_app.py --server.port 8501 --server.headless true
```

Required env:
- `OPENAI_API_KEY` (required for default provider)

Optional env:
- `MODEL_PROVIDER=openai|gemini` (default `openai`)
- `OPENAI_MODEL_NAME` (default `gpt-5-nano`)
- `GEMINI_API_KEY`
- `GEMINI_MODEL_NAME` (default `gemini-2.5-flash`)
- `MODEL_NAME` (hard override)
- `EMBEDDING_MODEL` (default `text-embedding-ada-002`)
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` (optional Qdrant RAG)
- `RAG_BACKEND=auto|qdrant|faiss` (default `auto`)
- `RAG_TOP_K` (default `3`)

Run tests:
```bash
python -m pytest tests/ -v
```

Quick validation:
```bash
python -c "from app.pipeline import pipeline; print('OK')"
```

## 3) End-to-end flow
Graph (from `app/pipeline.py`):
1. `ingest`
2. `safety_override` (first-pass risk, abuse, coercive control screening)
3. branch:
   - high risk -> `safety_response` -> `save_summary` -> END
   - else -> `context_modifier`
4. `context_modifier` (session framing: ordinary conflict, trust breach, possible abuse, etc.)
5. `phase_manager` (adaptive phase tracking with weighted readiness evidence)
6. `turn_router` (select turn_mode per message, independent of phase)
7. `specialist_orchestrator` (decide run_* flags using phase policy + turn_mode + safety + coaching readiness)
8. dispatch routes:
   - `intake_slot_fill` -> `intake_branch` (`smart_intake_agent`) -> (`intake_feedback` or `save_summary`)
   - otherwise -> support chain
9. support chain:
   - `rag_retrieval`
   - `combined_understanding`
   - `parallel_specialists`
   - `cultural_adapter`
   - `formulate_response`
   - `save_summary`

Important behavior:
- Safety override runs BEFORE all other routing.
- Phase manager uses weighted readiness evidence; hard caps trigger review, not forced advancement.
- Turn router decides what each message needs, independent of phase.
- Specialist orchestrator replaces old hard phase locks with policy-driven selection.
- Temporary fallback allows later-phase users to receive early-phase containment without phase reset.

## 4) Core state model
Defined in `app/models.py`.

Top-level state: `AppState`
- `turn: TurnState` (per-turn inputs, routing, risk, outputs, turn_mode, safety_flags)
- `case: CaseState` (category, slots, readiness, context_modifier, coaching_eligible)
- `profile: UserProfile` (culture/profile facts)
- `meta: SystemMeta` (turn_id/timestamps/latency)
- `therapy: TherapyPhaseState` (phase 1..5, phase_confidence, transition_decision, temporary_fallback)
- `conversation_history`

Key adaptive fields:
- `turn.turn_mode` / `turn.turn_mode_reason`: what this turn actually needs
- `turn.safety_override_triggered` / `turn.safety_flags`: safety state
- `case.context_modifier`: session framing (ordinary_conflict, possible_abuse, etc.)
- `case.coaching_eligible` / `case.coaching_eligibility_reason`: readiness gate
- `case.soft_signals_detected` / `case.milestones_completed`: evidence for progression
- `therapy.phase_transition_decision`: stay / advance / temporary_fallback / regress / review_needed
- `therapy.temporary_fallback`: containment mode without phase reset

Compatibility layer:
- `AppState` exposes flattened properties (`state.text`, `state.problem_category`, etc.) so old nodes continue working.
- `model_dump()` emits both nested + flattened keys for UI/persistence.

## 5) Therapy phase engine
Configured in `app/config.py` and managed by `app/agents/phase_manager.py`.

Phases:
1. Assessment
2. Understanding self/partner
3. Communication/conflict skills
4. Trust/emotional closeness
5. Stabilization/prevention

Advancement logic (adaptive):
- Weighted readiness score from milestones (55%), soft signals (30%), turn count (15%)
- Must meet `min_turns` AND readiness >= 0.65 to advance
- Hard cap (min_turns * 3) triggers `review_needed`, NOT forced advancement
- Supports: stay, advance, temporary_fallback, regress, review_needed

Phase policies (from `PHASE_POLICIES` in config.py):
- Each phase defines preferred, allowed-but-limited, and blocked response modes
- Policies are enforced by specialist_orchestrator, not by hard code blocks

Milestones are evaluated by LLM (`_evaluate_milestones_with_llm`) against recent history + slots.
Soft readiness signals are detected from user text (reflects on own role, considers partner, etc.).

## 6) Routing and modality control
### Safety override (`app/agents/risk_guard.py`)
Runs FIRST in the pipeline. Detects:
- Self-harm, harm to others, violence, threats
- Coercive control / abuse indicators
- Child safety concerns
- Severe emotional escalation
- Psychiatric red flags

Sets: `safety_override_triggered`, `safety_flags`, `risk_level`, `risk_type`, `risk_action`,
`allowed_actions`, `must_ask`, `must_refuse`, `escalate`.

### Context modifier (`app/agents/context_modifier.py`)
Classifies session framing:
- `ordinary_conflict` | `repair_after_breach` | `high_escalation` | `possible_abuse` |
  `separation_or_breakup` | `one_partner_unavailable` | `individual_reflection_mode`

### Turn router (`app/agents/turn_router.py`)
Selects `turn_mode` per message:
- `safety_check` | `empathy_containment` | `clarification` | `intake_slot_fill` |
  `psychoeducation` | `communication_coaching` | `trust_repair` | `closeness_building` |
  `maintenance_review` | `progress_reflection`

Phase biases which modes are preferred, but doesn't fully determine the turn.

### Specialist orchestrator (`app/agents/specialist_orchestrator.py`)
Replaces old `_support_entry_node()`. Decides run_* flags using:
- Phase policy (preferred/allowed/blocked modes)
- Turn mode
- Safety flags and context modifier
- Coaching readiness (emotional intensity + context completeness)
- Category modalities

## 7) Safety layer
Implemented in `app/agents/risk_guard.py` (renamed from `risk_guard` to `safety_override`).

High risk: forces `escalate_to_human`, routes to `safety_response`.
Medium risk: forces safety check question.
Coercive control: blocks couples exercises, prioritizes individual safety.
Psychiatric red flags: recommends professional psychiatric support.
Severe escalation: asks about immediate safety.

Note:
- `app/agents/safety_gate.py` exists but is not in active pipeline (legacy).

## 8) Intake subsystem
Main active intake node: `app/agents/smart_intake.py`

Capabilities:
- greeting handling
- slot extraction from current + recent context
- distress-aware intake responses
- help-now detection
- transition signaling (`intake_to_support`)
- profile note synthesis from slots

Intake is triggered when `turn_mode == intake_slot_fill` (from turn_router),
not by hard phase locks.

Slot schema and prompts live in config:
- `SLOTS_GENERIC`
- `SLOT_QUESTIONS`
- `SLOTS_REQUIRED_BY_CATEGORY`

Legacy intake files still present:
- `app/agents/intake.py`
- `app/agents/slot_intake.py`

## 9) Understanding + specialist generation
### Understanding
`app/agents/understanding.py` (`combined_understanding`):
- one LLM call for emotion + sentiment + problem category
- keyword fallback if model returns weak category
- can skip LLM when category already known and no emotion needed

### Parallel specialists
`app/agents/specialists.py` executes in parallel:
- emotion (`emotion_agent_compute`)
- coach (`coach_agent_compute`)
- growth (`growth_agent_compute`)
- psychoeducation (`psychoeducation_agent_compute`)
- pattern (`pattern_agent_compute`)

It merges results back into `state.turn` using `model_copy(update=...)` for LangGraph-safe merge behavior.

### Specialist intent
- `emotion.py`: reflective/validating text with confidence
- `coach.py`: practical tailored solution; adapts to `response_style`; avoids repeating prior advice
- `growth.py`: SMART goal + milestones + obstacle
- `psychoeducation.py`: explains why pattern is happening
- `pattern.py`: names/reframes recurring cycle

## 10) Response composition
Primary composer: `app/agents/response.py` (`formulate_response`).

What it does:
1. select empathy/advice components (confidence-aware)
2. enforce safety constraints (suppress advice when must_refuse is active)
3. choose dialogue action: ASK_ONE_QUESTION / RESPOND_AND_OPTIONAL_QUESTION / RESPOND_ONLY
4. generate final response with turn_mode-appropriate tone and banned phrase rules
5. include phase context + safety notes in prompt (`get_phase_context_for_prompt`)
6. adapt tone for context_modifier (abuse -> no couples exercises, separation -> no forced reconciliation)

Important guard:
- if smart intake already set final response with action `TRANSITION_TO_SUPPORT` or `ASK_ONE_QUESTION`, composer preserves it.

## 11) UI behavior
`app/ui/streamlit_app.py`:
- session state for history + profile + slots + phase metrics + routing metadata
- submits pipeline work via `background_executor`
- applies result only when `turn_id` matches active turn
- sidebar shows phase progress, milestones, adaptive routing info (turn_mode, coaching eligibility, safety flags, context modifier), slot info
- supports reset (new `session_id`)

## 12) Persistence
`app/agents/persistence.py` writes SQLite tables:
- `conversation_summary` (one row per turn, includes `decision_metadata` JSON blob)
- `therapy_sessions` (phase/session-level state, includes adaptive fields in `phase_data`)

Includes schema migration for older DBs (`PRAGMA table_info` + `ALTER TABLE`).

Saved data includes:
- turn metadata, category, slots, risk, action type, user text, emotion
- therapy phase/session fields (`session_id`, `therapy_phase`, milestones/history)
- `decision_metadata`: turn_mode, turn_mode_reason, safety_override_triggered, safety_flags,
  context_modifier, readiness_score, coaching_eligible, phase_transition_decision, etc.

## 13) LLM provider layer
`app/llm/providers.py`:
- unified `ask_model(...)`
- dispatches to OpenAI or Gemini
- OpenAI client timeout = 90s
- Gemini import/config done lazily

## 14) RAG implementation
`app/agents/rag.py`:
- optional Qdrant retrieval when `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` are configured
- query embeddings use OpenAI via `app/llm/embeddings.py` and `EMBEDDING_MODEL`
- local FAISS `IndexFlatL2` fallback with deterministic pseudo-embeddings (`embed_text_local`)
- only 2 placeholder docs exist in the fallback index
- query built from category + slots + turn text
- `scripts/ingest_qdrant.py` seeds or updates Qdrant using matching OpenAI embeddings

Implication:
- Qdrant can provide production-style semantic retrieval once the collection is populated with real counseling content.
- The FAISS fallback remains scaffolding/demo quality.

## 15) Active modules
All modules in `app/agents/` are actively used in the graph path:
- ingest, safety_override (risk_guard), context_modifier, phase_manager, turn_router,
  specialist_orchestrator, smart_intake, intake_feedback, rag_retrieval, combined_understanding,
  parallel_specialists (specialists), cultural_adapter, formulate_response (response), persistence
- Individual specialist agents: emotion, coach, growth, psychoeducation, pattern
- Auxiliary: transcription, self_eval (optional, disabled by default)

Legacy modules (triage_router, smalltalk, profile, need, scheduler, dialogue_manager,
response_selector, classification, orchestrator, slot_intake, intake, safety_gate,
therapy_specialist) were removed as they were not used in the active pipeline.

## 16) Known caveats and implementation notes
- Some source/docs show mojibake for Arabic text due encoding handling in current environment.
- `transcription.py` uses old `openai.Audio.transcribe` style and is not integrated with the newer OpenAI client pattern used elsewhere.
- `openai` package is used by both providers module and transcription; any SDK major changes can break this path.

## 17) Complete file index (what each file does)
### Root
- `AGENTS.md`: environment/task instructions and caveats.
- `ONBOARDING.md`: codebase onboarding (quick start, architecture, data, env).
- `README.md`: project summary and basic run instructions.
- `workflow.md`: high-level therapy pipeline and phase docs.
- `requirements.txt`: Python dependencies.
- `run_app.py`: thin launcher calling `run_streamlit_app()`; prefer `streamlit run app/ui/streamlit_app.py`.

### app/
- `app/__init__.py`: global logging setup.
- `app/config.py`: env config, therapy definitions, phase policies, cue dictionaries, slot templates, safety keyword sets.
- `app/models.py`: Pydantic state schema + compatibility flattening.
- `app/pipeline.py`: LangGraph construction, routing functions, background run helper.
- `app/utils.py`: intake/profile helper functions.

### app/llm/
- `app/llm/__init__.py`: exports `ask_model`.
- `app/llm/embeddings.py`: OpenAI embedding helper for Qdrant retrieval and ingestion.
- `app/llm/providers.py`: OpenAI/Gemini wrappers.

### app/ui/
- `app/ui/__init__.py`: package marker.
- `app/ui/streamlit_app.py`: Streamlit chat app and async integration.

### app/agents/
- `app/agents/__init__.py`: exports agent nodes.
- `app/agents/ingest.py`: text/audio ingest pre-step.
- `app/agents/transcription.py`: whisper transcription + cleanup.
- `app/agents/risk_guard.py`: safety override layer (first-pass risk, abuse, coercive control screening).
- `app/agents/context_modifier.py`: session framing classifier (ordinary conflict, trust breach, abuse, etc.).
- `app/agents/phase_manager.py`: adaptive phase progression with weighted readiness evidence.
- `app/agents/turn_router.py`: per-turn mode selection (containment, coaching, psychoeducation, etc.).
- `app/agents/specialist_orchestrator.py`: policy-driven specialist selection.
- `app/agents/smart_intake.py`: active intake logic.
- `app/agents/intake_feedback.py`: post-intake therapist feedback node.
- `app/agents/understanding.py`: combined emotion/sentiment/category analysis.
- `app/agents/specialists.py`: parallel specialist executor.
- `app/agents/emotion.py`: empathy generation.
- `app/agents/coach.py`: practical tailored coaching output.
- `app/agents/growth.py`: long-term SMART growth guidance.
- `app/agents/psychoeducation.py`: explanatory insight on relationship dynamics.
- `app/agents/pattern.py`: cycle naming/reframing.
- `app/agents/cultural_adapter.py`: cultural phrasing adaptation.
- `app/agents/response.py`: unified response composer with safety enforcement.
- `app/agents/self_eval.py`: optional post-response checker (disabled by default).
- `app/agents/rag.py`: optional Qdrant retrieval with local FAISS fallback.
- `app/agents/persistence.py`: SQLite persistence with decision metadata.

### scripts/
- `scripts/ingest_qdrant.py`: ingest JSONL chunks or seed fallback docs into Qdrant.

### tests/
- `tests/conftest.py`: shared fixtures for adaptive workflow tests.
- `tests/test_phase_manager.py`: weighted progression, no hard-cap auto-advance, temporary fallback, regression.
- `tests/test_safety_override.py`: high risk, coercive control, violence/abuse, child safety, psychiatric flags.
- `tests/test_turn_router.py`: distress containment, practical asks, flooded turns, stable phase oscillation.
- `tests/test_orchestrator.py`: specialist eligibility from phase + readiness + context modifier.
- `tests/test_pipeline_scenarios.py`: end-to-end state transitions for acceptance scenarios.

## 18) If you are a new coding agent, start here
1. Read: `app/pipeline.py`, `app/models.py`, `app/config.py`, `app/agents/turn_router.py`, `app/agents/specialist_orchestrator.py`.
2. Confirm branch assumptions: whether you are editing active graph path vs legacy modules.
3. If changing behavior, update both:
   - implementation in active nodes
   - docs: `workflow.md` and this file if architecture changes.
4. Validate by running `python -m pytest tests/ -v` and one manual Streamlit run (`python -m streamlit run app/ui/streamlit_app.py`).
