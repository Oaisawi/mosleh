# Mosleh – Couples Counseling Assistant
Multi-agent conversational flow with emotion detection, coaching, and cultural guidance. Adaptive 5-phase therapy pipeline with safety-first routing.

## Project structure

```
Capstone2/
├── app/
│   ├── __init__.py
│   ├── config.py              # Env, API keys, model names, phase policies
│   ├── models.py              # AppState (Pydantic)
│   ├── utils.py               # Intake/profile helpers
│   ├── pipeline.py            # LangGraph pipeline and runners
│   ├── llm/
│   │   ├── __init__.py
│   │   └── providers.py       # OpenAI / Gemini chat
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ingest.py          # Text/audio ingest pre-step
│   │   ├── transcription.py   # Whisper transcription
│   │   ├── risk_guard.py      # Safety override (abuse, coercive control)
│   │   ├── context_modifier.py # Session framing classifier
│   │   ├── phase_manager.py   # Adaptive phase progression
│   │   ├── turn_router.py     # Per-turn mode selection
│   │   ├── specialist_orchestrator.py # Policy-driven specialist selection
│   │   ├── smart_intake.py    # Context-aware intake
│   │   ├── intake_feedback.py # Post-intake therapist feedback
│   │   ├── emotion.py         # Empathy generation
│   │   ├── coach.py           # Practical coaching
│   │   ├── growth.py          # Long-term growth guidance
│   │   ├── psychoeducation.py # Insight on relationship dynamics
│   │   ├── pattern.py         # Cycle naming/reframing
│   │   ├── specialists.py     # Parallel specialist executor
│   │   ├── understanding.py   # Combined emotion/sentiment/category
│   │   ├── cultural_adapter.py # Cultural phrasing adaptation
│   │   ├── rag.py             # Qdrant RAG with FAISS fallback
│   │   ├── response.py        # Final response composer
│   │   └── persistence.py     # SQLite persistence
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py   # Chat UI (Streamlit entry target)
├── run_app.py                 # Calls run_streamlit_app(); prefer Streamlit CLI below
├── requirements.txt
└── tests/                     # Pytest suite
```

## Run the app

From the project root:

```bash
pip install -r requirements.txt
python -m streamlit run app/ui/streamlit_app.py
```

Optional headless:

```bash
streamlit run app/ui/streamlit_app.py --server.port 8501 --server.headless true
```

## Environment

- `OPENAI_API_KEY` – required for default provider
- `MODEL_PROVIDER` – `openai` (default) or `gemini`
- `OPENAI_MODEL_NAME` / `GEMINI_MODEL_NAME` – Model names
- `GEMINI_API_KEY` – required if using Gemini
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` – optional Qdrant RAG settings
- `RAG_BACKEND` – `auto`, `qdrant`, or `faiss`; use `qdrant` with collection `moslehai_kb_v3`
- `RAG_TOP_K` – number of retrieved knowledge chunks, default `3`

See [ONBOARDING.md](ONBOARDING.md) for architecture, SQLite, and fuller setup notes.
