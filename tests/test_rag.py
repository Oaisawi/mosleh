"""Tests for optional Qdrant RAG retrieval."""

from app.agents import rag


def test_rag_falls_back_to_faiss_when_qdrant_disabled(base_state, monkeypatch):
    monkeypatch.setattr(rag, "RAG_BACKEND", "faiss")
    base_state.turn.run_rag = True
    base_state.turn.text = "we are anxious and stressed"

    rag.rag_retrieval(base_state)

    assert base_state.retrieved_info
    assert "stress" in base_state.retrieved_info.lower()


def test_rag_uses_qdrant_payload_text(base_state, monkeypatch):
    calls = {}

    def fake_request(method, url, payload):
        calls["method"] = method
        calls["url"] = url
        calls["payload"] = payload
        return {
            "result": {
                "points": [
                    {"payload": {"text": "grounded counseling chunk"}},
                    {"payload": {"content": "second retrieved chunk"}},
                ]
            }
        }

    monkeypatch.setattr(rag, "RAG_BACKEND", "qdrant")
    monkeypatch.setattr(rag, "QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setattr(rag, "QDRANT_API_KEY", "test-key")
    monkeypatch.setattr(rag, "QDRANT_COLLECTION", "mosleh_test")
    monkeypatch.setattr(rag, "RAG_TOP_K", 2)
    monkeypatch.setattr(rag, "embed_text_openai", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(rag, "_request_json", fake_request)

    base_state.turn.run_rag = True
    base_state.turn.text = "how can we rebuild trust"

    rag.rag_retrieval(base_state)

    assert calls == {
        "method": "POST",
        "url": "https://example.qdrant.io/collections/mosleh_test/points/query",
        "payload": {
            "query": [0.1, 0.2, 0.3],
            "limit": 2,
            "with_payload": True,
        },
    }
    assert base_state.retrieved_info == "grounded counseling chunk\nsecond retrieved chunk"


def test_rag_handles_qdrant_result_list_shape(base_state, monkeypatch):
    monkeypatch.setattr(rag, "RAG_BACKEND", "qdrant")
    monkeypatch.setattr(rag, "QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setattr(rag, "QDRANT_API_KEY", "test-key")
    monkeypatch.setattr(rag, "QDRANT_COLLECTION", "mosleh_test")
    monkeypatch.setattr(rag, "embed_text_openai", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        rag,
        "_request_json",
        lambda method, url, payload: {
            "result": [
                {"payload": {"chunk": "legacy shaped chunk"}},
            ]
        },
    )

    base_state.turn.run_rag = True
    base_state.turn.text = "how can we rebuild trust"

    rag.rag_retrieval(base_state)

    assert base_state.retrieved_info == "legacy shaped chunk"


def test_rag_falls_back_to_faiss_when_qdrant_errors(base_state, monkeypatch):
    monkeypatch.setattr(rag, "RAG_BACKEND", "qdrant")
    monkeypatch.setattr(rag, "QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setattr(rag, "QDRANT_API_KEY", "test-key")
    monkeypatch.setattr(rag, "QDRANT_COLLECTION", "mosleh_test")
    monkeypatch.setattr(rag, "embed_text_openai", lambda text: (_ for _ in ()).throw(RuntimeError("offline")))

    base_state.turn.run_rag = True
    base_state.turn.text = "we are anxious and stressed"

    rag.rag_retrieval(base_state)

    assert base_state.retrieved_info
    assert "stress" in base_state.retrieved_info.lower()
