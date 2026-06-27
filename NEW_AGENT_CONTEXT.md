# New Agent Project Context: Mosleh AI (Capstone2)

## 1) What this project is
Mosleh AI is a single-service Streamlit + LangGraph couples-counseling assistant.
It runs as one Python app with:
- no external DB server (SQLite file: `conversation_logs.db`)
- optional Qdrant vector DB for grounded RAG, with in-process FAISS fallback
- OpenAI (default) or Gemini via env switch
- a 5-phase therapy workflow enforced in routing

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
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION=moslehai_kb_v3` (optional Qdrant RAG)
- `RAG_BACKEND=auto|qdrant|faiss` (use `qdrant` for the populated cloud KB)
- `RAG_TOP_K` (default `3`)

Quick validation (no formal tests configured):
```bash
python3 -c "import importlib, pkgutil; [importlib.import_module(m) for _,m,_ in pkgutil.walk_packages(['app'], prefix='app.')]"
```

## 3) End-to-end flow
Graph (from `app/pipeline.py`):
1. `ingest`
2. `triage_router`
3. `phase_manager`
4. `risk_guard`
5. branch:
- high risk -> `safety_response` -> `save_summary` -> END
- else -> `dispatch`
6. dispatch routes:
- `intake_needed` -> `intake_branch` (`smart_intake_agent`) -> (`support_entry` or `save_summary`)
- otherwise -> `support_entry`
7. support chain:
- `rag_retrieval`
- `combined_understanding`
- `parallel_specialists`
- `cultural_adapter`
- `formulate_response`
- `save_summary`

Important behavior:
- Phase 1 does not transition from intake to support directly.
- `phase_manager` controls phase advancement.
- `support_entry` is the authority that sets `run_*` flags.

## 4) Core state model
Defined in `app/models.py`.

Top-level state: `AppState`
- `turn: TurnState` (per-turn inputs, routing, risk, outputs)
- `case: CaseState` (category, slots, readiness, intake progress)
- `profile: UserProfile` (culture/profile facts)
- `meta: SystemMeta` (turn_id/timestamps/latency)
- `therapy: TherapyPhaseState` (phase 1..5 tracking)
- `conversation_history`

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

Advancement logic (`_should_advance_phase`):
- must meet `min_turns`
- must reach >=75% milestones
- hard cap fallback: `min_turns * 3`

Milestones are evaluated by LLM (`_evaluate_milestones_with_llm`) against recent history + slots.

## 6) Routing and modality control
### `triage_router`
Sets:
- `turn_type`
- `user_intent`
- `emotional_intensity`
- `response_style`
- `needs_rag`

It is phase-aware and prioritizes:
1. high risk
2. understanding/pattern asks
3. distress handling
4. dissatisfaction cues
5. intent classification

### `support_entry` (`app/pipeline.py`)
Calculates readiness and enforces which specialist agents run.
- outputs `run_emotion`, `run_coach`, `run_growth`, `run_psychoeducation`, `run_pattern`, `run_rag`
- applies strong phase gating (coach/growth unavailable in early phases)
- writes summary plan into `state.case.plan`

## 7) Safety layer
Implemented in `app/agents/risk_guard.py`.

`risk_guard` sets:
- `risk_level`, `risk_type`, `risk_action`
- safety gate fields: `allowed_actions`, `must_ask`, `must_refuse`, `escalate`

High risk:
- forces `escalate_to_human`
- routes to `safety_response`

Medium risk:
- forces safety check question

Note:
- `app/agents/safety_gate.py` exists but is not in active pipeline (legacy/separate node implementation).

## 8) Intake subsystem
Main active intake node: `app/agents/smart_intake.py`

Capabilities:
- greeting handling
- slot extraction from current + recent context
- distress-aware intake responses
- help-now detection
- transition signaling (`intake_to_support`)
- profile note synthesis from slots

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

This file absorbs old behavior from:
- `dialogue_manager.py`
- `response_selector.py`

What it does:
1. select empathy/advice components (confidence-aware)
2. choose dialogue action:
- `ASK_ONE_QUESTION`
- `RESPOND_AND_OPTIONAL_QUESTION`
- `RESPOND_ONLY`
3. generate final response with strict tone/style and banned phrase rules
4. include phase context in prompt (`get_phase_context_for_prompt`)

Important guard:
- if smart intake already set final response with action `TRANSITION_TO_SUPPORT` or `ASK_ONE_QUESTION`, composer preserves it.

## 11) UI behavior
`app/ui/streamlit_app.py`:
- session state for history + profile + slots + phase metrics
- submits pipeline work via `background_executor`
- applies result only when `turn_id` matches active turn
- sidebar shows phase progress, milestones, routing info, slot info
- supports reset (new `session_id`)

## 12) Persistence
`app/agents/persistence.py` writes SQLite tables:
- `conversation_summary` (one row per turn)
- `therapy_sessions` (phase/session-level state)

Includes schema migration for older DBs (`PRAGMA table_info` + `ALTER TABLE`).

Saved data includes:
- turn metadata, category, slots, risk, action type, user text, emotion
- therapy phase/session fields (`session_id`, `therapy_phase`, milestones/history)

## 13) LLM provider layer
`app/llm/providers.py`:
- unified `ask_model(...)`
- dispatches to OpenAI or Gemini
- OpenAI client timeout = 90s
- Gemini import/config done lazily

## 14) RAG implementation
`app/agents/rag.py`:
- Qdrant REST retrieval when `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` are configured
- query built from category + slots + turn text
- OpenAI embeddings via `app/llm/embeddings.py`
- FAISS `IndexFlatL2` fallback with deterministic pseudo-embeddings

Implication:
- use `RAG_BACKEND=qdrant` and `QDRANT_COLLECTION=moslehai_kb_v3` for the populated cloud KB.

## 15) Active vs legacy modules
Active in graph path:
- ingest, triage_router, phase_manager, risk_guard, smart_intake, support_entry,
  rag_retrieval, combined_understanding, parallel_specialists, cultural_adapter,
  formulate_response, persistence

Present but not graph-wired (legacy/aux):
- `smalltalk.py`, `profile.py`, `need.py`, `scheduler.py`, `dialogue_manager.py`,
  `response_selector.py`, `classification.py`, `orchestrator.py`, `slot_intake.py`,
  `intake.py`, `safety_gate.py`, `therapy_specialist.py`

## 16) Known caveats and implementation notes
- Some source/docs show mojibake for Arabic text due encoding handling in current environment.
- `transcription.py` uses old `openai.Audio.transcribe` style and is not integrated with the newer OpenAI client pattern used elsewhere.
- `openai` package is used by both providers module and transcription; any SDK major changes can break this path.
- no test suite/linter pipeline in repo.

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
- `app/config.py`: env config, therapy definitions, cue dictionaries, slot templates.
- `app/models.py`: Pydantic state schema + compatibility flattening.
- `app/pipeline.py`: LangGraph construction, routing functions, background run helper.
- `app/utils.py`: intake/profile helper functions.

### app/llm/
- `app/llm/__init__.py`: exports `ask_model`.
- `app/llm/providers.py`: OpenAI/Gemini wrappers.

### app/ui/
- `app/ui/__init__.py`: package marker.
- `app/ui/streamlit_app.py`: Streamlit chat app and async integration.

### app/agents/
- `app/agents/__init__.py`: exports agent nodes.
- `app/agents/ingest.py`: text/audio ingest pre-step.
- `app/agents/transcription.py`: whisper transcription + cleanup.
- `app/agents/triage_router.py`: turn type + response style routing.
- `app/agents/phase_manager.py`: phase progression/milestones.
- `app/agents/risk_guard.py`: risk classification + safety action.
- `app/agents/safety_gate.py`: legacy separate safety gate.
- `app/agents/smart_intake.py`: active intake logic.
- `app/agents/slot_intake.py`: legacy slot intake node.
- `app/agents/intake.py`: legacy intake + info collector.
- `app/agents/understanding.py`: combined emotion/sentiment/category analysis.
- `app/agents/specialists.py`: parallel specialist executor.
- `app/agents/emotion.py`: empathy generation.
- `app/agents/coach.py`: practical tailored coaching output.
- `app/agents/growth.py`: long-term SMART growth guidance.
- `app/agents/psychoeducation.py`: explanatory insight on relationship dynamics.
- `app/agents/pattern.py`: cycle naming/reframing.
- `app/agents/cultural_adapter.py`: cultural phrasing adaptation.
- `app/agents/therapy_specialist.py`: legacy cultural adapter variant.
- `app/agents/response.py`: current unified composer.
- `app/agents/response_selector.py`: legacy selection module.
- `app/agents/dialogue_manager.py`: legacy dialogue action module.
- `app/agents/classification.py`: legacy category classifier.
- `app/agents/orchestrator.py`: legacy plan text builder.
- `app/agents/need.py`: legacy need flags logic.
- `app/agents/scheduler.py`: legacy run-flag scheduler.
- `app/agents/self_eval.py`: optional post-response checker (disabled by default).
- `app/agents/profile.py`: profile updater + intro parallel runner.
- `app/agents/smalltalk.py`: greeting response (legacy branch).
- `app/agents/rag.py`: Qdrant RAG with local FAISS fallback.
- `app/agents/persistence.py`: SQLite persistence.

## 18) If you are a new coding agent, start here
1. Read: `app/pipeline.py`, `app/models.py`, `app/config.py`, `app/agents/response.py`, `app/agents/smart_intake.py`.
2. Confirm branch assumptions: whether you are editing active graph path vs legacy modules.
3. If changing behavior, update both:
- implementation in active nodes
- docs: `workflow.md` and this file if architecture changes.
4. Validate by import check and one manual Streamlit run.
