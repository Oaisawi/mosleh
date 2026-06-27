"""Ingest counseling knowledge chunks into Qdrant.

Vectors must be created with the same OpenAI embedding model used at query time.
Defaults use app.config.EMBEDDING_MODEL and app.config.EMBEDDING_DIM.
"""
import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.agents.rag import DOC_TEXTS
from app.config import (
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from app.llm.embeddings import embed_text_openai

logger = logging.getLogger(__name__)


def _load_chunks(path: Path | None) -> List[Dict[str, str]]:
    if path is None:
        return [{"id": f"seed-{idx}", "text": text} for idx, text in enumerate(DOC_TEXTS)]

    chunks: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            text = str(item.get("text") or item.get("content") or "").strip()
            if not text:
                raise ValueError(f"Missing text/content at {path}:{line_number}")
            raw_id = str(item.get("id") or f"{path.name}:{line_number}")
            chunks.append({"id": raw_id, "text": text})
    return chunks


def _point_id(raw_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        logger.info("Creating Qdrant collection %s", QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def _points(chunks: Iterable[Dict[str, str]]) -> Iterable[PointStruct]:
    for chunk in chunks:
        text = chunk["text"]
        yield PointStruct(
            id=_point_id(chunk["id"]),
            vector=embed_text_openai(text),
            payload={"source_id": chunk["id"], "text": text},
        )


def ingest(path: Path | None = None) -> int:
    if not (QDRANT_URL and QDRANT_API_KEY and QDRANT_COLLECTION):
        raise RuntimeError("QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION are required.")

    chunks = _load_chunks(path)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    _ensure_collection(client)
    client.upsert(collection_name=QDRANT_COLLECTION, points=list(_points(chunks)))
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL chunks into Qdrant.")
    parser.add_argument(
        "--input",
        type=Path,
        help='Optional JSONL file with {"id": "...", "text": "..."} per line.',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    count = ingest(args.input)
    logger.info("Ingested %s chunks into %s", count, QDRANT_COLLECTION)


if __name__ == "__main__":
    main()
