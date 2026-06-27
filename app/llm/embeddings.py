"""Embedding helpers used by retrieval and ingestion."""
from typing import List, Optional

from openai import OpenAI

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY

EMBEDDING_TIMEOUT = 30.0


def embed_text_openai(text: str, model: Optional[str] = None) -> List[float]:
    """Create an OpenAI embedding for semantic retrieval."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=EMBEDDING_TIMEOUT)
    response = client.embeddings.create(
        model=model or EMBEDDING_MODEL,
        input=text,
    )
    return list(response.data[0].embedding)
