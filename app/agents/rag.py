"""RAG retrieval with optional Qdrant REST search and local FAISS fallback."""
from functools import lru_cache
import json
import logging
import urllib.error
import urllib.request

import faiss
import numpy as np

from app.config import (
    EMBED_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RAG_BACKEND,
    RAG_TOP_K,
)
from app.llm.embeddings import embed_text_openai
from app.logutil import ctx_from_state

logger = logging.getLogger(__name__)

QDRANT_TIMEOUT = 30.0

# Sample documents for the vector store
DOC_TEXTS = [
    "Tips for managing work-related stress: Prioritize tasks, take short breaks, and communicate with your manager about workload.",
    "Techniques to cope with anxiety include deep breathing, mindfulness meditation, and speaking with a trusted friend or counselor.",
]


def build_rag_query(state) -> str:
    """Build RAG query from case frame: category + slots + user goal, not raw user text."""
    category = (state.case.problem_category or "").strip()
    slots = state.case.slots_filled or {}
    goal = slots.get("desired_outcome", "") or slots.get("situation_summary", "")
    text = (state.turn.text or "").strip()
    parts = [p for p in [category, goal, text] if p]
    return " ".join(parts) if parts else text or "couples counseling support"


def embed_text_local(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Lightweight, deterministic embedding to avoid slow API calls on startup.
    Uses a seeded PRNG on the text for reproducibility.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


@lru_cache(maxsize=1)
def load_vector_resources():
    """Load or create the FAISS index for lightweight RAG without remote calls."""
    doc_embeddings = [embed_text_local(doc) for doc in DOC_TEXTS]
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.stack(doc_embeddings, dtype=np.float32))
    index_to_doc = {i: DOC_TEXTS[i] for i in range(len(DOC_TEXTS))}
    return index, index_to_doc


def _qdrant_is_configured() -> bool:
    return bool(QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION)


def _should_use_qdrant() -> bool:
    if RAG_BACKEND == "faiss":
        return False
    if RAG_BACKEND == "qdrant":
        return True
    return _qdrant_is_configured()


def _payload_to_text(payload: dict) -> str:
    for key in ("text", "content", "chunk"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _retrieve_faiss(query: str, k: int = 2) -> str:
    index, index_to_doc = load_vector_resources()
    query_embedding = embed_text_local(query)
    _, indices = index.search(np.array([query_embedding], dtype=np.float32), k=k)
    retrieved_docs = [index_to_doc[idx] for idx in indices[0] if idx != -1]
    return "\n".join(retrieved_docs)


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    """Call a JSON HTTP endpoint with stdlib urllib.

    The qdrant-client/httpx path can hang in some local Windows environments;
    urllib has been reliable for this project and keeps runtime imports light.
    """
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY or "",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=QDRANT_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant request failed: {exc.code} {detail}") from exc


def _qdrant_points(query_embedding):
    """Search Qdrant through the REST query API."""
    url = (
        f"{QDRANT_URL.rstrip('/')}/collections/"
        f"{QDRANT_COLLECTION}/points/query"
    )
    response = _request_json(
        "POST",
        url,
        {
            "query": query_embedding,
            "limit": RAG_TOP_K,
            "with_payload": True,
        },
    )
    result = response.get("result", {})
    if isinstance(result, dict):
        return result.get("points", [])
    if isinstance(result, list):
        return result
    return []


def _retrieve_qdrant(query: str) -> str:
    if not _qdrant_is_configured():
        raise RuntimeError("Qdrant is not configured.")
    query_embedding = embed_text_openai(query)
    points = _qdrant_points(query_embedding)
    snippets = []
    for point in points or []:
        payload = point.get("payload", {}) if isinstance(point, dict) else getattr(point, "payload", None) or {}
        text = _payload_to_text(payload)
        if text:
            snippets.append(text)
    return "\n".join(snippets)


def rag_retrieval(state):
    """Retrieve relevant info from the knowledge base. Only when run_rag; query from case frame."""
    if not getattr(state.turn, "run_rag", True):
        logger.info("rag_retrieval %s skip reason=run_rag_false", ctx_from_state(state))
        return {}
    if not state.text and not build_rag_query(state):
        logger.info("rag_retrieval %s skip reason=empty_query", ctx_from_state(state))
        return {}
    query = build_rag_query(state)
    logger.info("rag_retrieval %s run query_len=%s", ctx_from_state(state), len(query or ""))

    context = ""
    if _should_use_qdrant():
        try:
            context = _retrieve_qdrant(query)
            logger.info(
                "rag_retrieval %s backend=qdrant snippets=%s",
                ctx_from_state(state),
                len([line for line in context.splitlines() if line.strip()]),
            )
        except Exception as exc:
            logger.warning(
                "rag_retrieval %s qdrant_fallback reason=%s",
                ctx_from_state(state),
                exc,
            )
            context = _retrieve_faiss(query)
    else:
        context = _retrieve_faiss(query)
        logger.info("rag_retrieval %s backend=faiss", ctx_from_state(state))

    state.retrieved_info = context
    return {}
