# rag_test_final_multifilter.py
# Local-first RAG test script for MoslehAI KB:
# Run:
#   python .\rag_test_final_multifilter.py --kb .\kb_min_v1.jsonl --generate --min_score 0.35 --top_k 4

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss
from openai import OpenAI


# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def detect_language(text: str) -> str:
    """Very simple language detection for routing retrieval."""
    return "ar" if re.search(r"[\u0600-\u06FF]", text or "") else "en"


def safe_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_alignment_tags(meta: Dict[str, Any]) -> List[str]:
    """
    Your KB entries often look like:
      {"id":..., "source_type":..., "language":..., "topic":..., "meta": {"alignment_tags":[...]}, "text":...}

    So tags may live in:
      meta["meta"]["alignment_tags"]   (preferred)
    or sometimes directly:
      meta["alignment_tags"]
    """
    tags = []
    inner = meta.get("meta")
    if isinstance(inner, dict):
        tags = safe_list(inner.get("alignment_tags"))
    if not tags:
        tags = safe_list(meta.get("alignment_tags"))
    # normalize
    return [str(t).strip() for t in tags if str(t).strip()]


def infer_topic(text: str) -> str:
    """
    Lightweight topic guesser for convenience only.
    If you want strict behavior, always provide [topic=...].
    """
    t = (text or "").lower()

    # finances
    if any(w in t for w in ["money", "salary", "debt", "bills", "rent", "loan", "financ", "budget"]):
        return "finances"

    # trust
    if any(w in t for w in ["cheat", "cheating", "affair", "trust", "lying", "lied", "suspicious", "phone", "messages"]):
        return "trust"

    # family interference
    if any(w in t for w in ["mother", "father", "in-law", "in law", "parents", "family interfering", "his family", "her family"]):
        return "family_interference"

    # intimacy
    if any(w in t for w in ["intimacy", "close", "closeness", "roommates", "sex", "affection", "romance", "cold"]):
        return "intimacy"

    # respect
    if any(w in t for w in ["respect", "insult", "humiliate", "mock", "name-calling", "yell", "yelling", "sarcasm", "contempt"]):
        return "respect"

    # SAFETY / DANGER (High Priority)
    if any(w in t for w in ["hit", "beat", "harm", "kill", "suicide", "abuse", "danger", "911", "emergency", "blood", "threat", "scared", "fear", "hurt"]):
        return "safety"

    # default
    return "communication"


def parse_query_filters(q: str) -> Tuple[Dict[str, Any], str]:
    """
    Parses a leading filter block:
      [topic=communication source=book tags=a,b,c lang=en]
    Returns (filters, remaining_query).
    """
    filters: Dict[str, Any] = {
        "lang": None,
        "topic": None,
        "source_type": None,
        "tags": [],
    }

    s = q.strip()
    if not s.startswith("["):
        return filters, s

    m = re.match(r"^\[(.*?)\]\s*(.*)$", s)
    if not m:
        return filters, s

    inside = m.group(1).strip()
    rest = m.group(2).strip()

    # Split by spaces, but keep commas for tags list
    parts = [p.strip() for p in re.split(r"\s+", inside) if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()

        if k in ("lang", "language"):
            filters["lang"] = v.lower()
        elif k == "topic":
            filters["topic"] = v.lower()
        elif k in ("source", "source_type"):
            filters["source_type"] = v.lower()
        elif k in ("tags", "alignment_tags", "tag"):
            tags = [t.strip() for t in v.split(",") if t.strip()]
            filters["tags"] = [t for t in tags]
        elif k in ("target", "audience", "for"):
            filters["target_audience"] = v.lower()
        # ignore unknown keys silently

    return filters, rest


# -----------------------------
# Data model
# -----------------------------
@dataclass
class KBItem:
    id: str
    text: str
    meta: Dict[str, Any]
    vec: Optional[np.ndarray] = None  # filled after embedding


def load_kb_jsonl(path: str) -> List[KBItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB file not found: {path}")

    items: List[KBItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}")

            _id = clean_text(str(obj.get("id", "")))
            text = clean_text(str(obj.get("text", "")))
            if not _id or not text:
                continue

            meta = {k: v for k, v in obj.items() if k != "text"}
            items.append(KBItem(id=_id, text=text, meta=meta))

    if not items:
        raise ValueError("KB loaded but empty. Check your JSONL has 'id' and 'text' per line.")

    return items


# -----------------------------
# Embeddings
# -----------------------------
def embed_openai(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    mat = np.vstack(vecs)
    return l2_normalize(mat)


def embed_items_in_batches(client: OpenAI, items: List[KBItem], emb_model: str, batch_size: int = 128) -> None:
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        mat = embed_openai(client, [it.text for it in batch], emb_model)
        for it, v in zip(batch, mat):
            it.vec = v


# -----------------------------
# Index building
# -----------------------------
def make_index_from_items(items: List[KBItem]) -> Optional[faiss.IndexFlatIP]:
    if not items:
        return None
    if items[0].vec is None:
        raise RuntimeError("Vectors missing; embed_items_in_batches must run first.")

    dim = int(items[0].vec.shape[0])
    index = faiss.IndexFlatIP(dim)
    mat = np.vstack([it.vec for it in items]).astype("float32")
    index.add(mat)
    return index


def build_indexes(client: OpenAI, items: List[KBItem], emb_model: str) -> Dict[str, Any]:
    """
    Prebuild indexes for:
      lang -> all
      lang -> topic -> all
      lang -> topic -> source_type

    Returns structure:
      indexes[lang]["all"] = (index, items_list)
      indexes[lang]["topic"][topic]["all"] = (index, items_list)
      indexes[lang]["topic"][topic]["source"][source_type] = (index, items_list)
    """
    # Split by language (we embed separately only for routing clarity; could embed all at once too)
    en_items = [it for it in items if it.meta.get("language") == "en"]
    ar_items = [it for it in items if it.meta.get("language") == "ar"]

    # Embed once per item
    if en_items:
        embed_items_in_batches(client, en_items, emb_model)
    if ar_items:
        embed_items_in_batches(client, ar_items, emb_model)

    indexes: Dict[str, Any] = {}
    for lang, lang_items in (("en", en_items), ("ar", ar_items)):
        indexes[lang] = {"all": None, "topic": {}}
        if not lang_items:
            continue

        # base index
        indexes[lang]["all"] = (make_index_from_items(lang_items), lang_items)

        # group by topic
        topics = sorted({str(it.meta.get("topic", "any")).strip() for it in lang_items if it.meta.get("topic")})
        for topic in topics:
            t_items = [it for it in lang_items if it.meta.get("topic") == topic]
            if not t_items:
                continue

            indexes[lang]["topic"].setdefault(topic, {"all": None, "source": {}})
            indexes[lang]["topic"][topic]["all"] = (make_index_from_items(t_items), t_items)

            # group by source_type under topic
            source_types = sorted({str(it.meta.get("source_type", "any")).strip() for it in t_items if it.meta.get("source_type")})
            for st in source_types:
                st_items = [it for it in t_items if it.meta.get("source_type") == st]
                if not st_items:
                    continue
                indexes[lang]["topic"][topic]["source"][st] = (make_index_from_items(st_items), st_items)

    return indexes


# -----------------------------
# Retrieval
# -----------------------------
def search_index(
    client: OpenAI,
    index: faiss.IndexFlatIP,
    items: List[KBItem],
    query: str,
    emb_model: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_vec = embed_openai(client, [query], emb_model).astype("float32")
    scores, idxs = index.search(q_vec, top_k)

    out = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
        if idx == -1:
            continue
        it = items[idx]
        out.append(
            {
                "rank": rank,
                "score": float(score),
                "id": it.id,
                "meta": it.meta,
                "text": it.text,
            }
        )
    return out


def search_index_with_vec(
    index: faiss.IndexFlatIP,
    items: List[KBItem],
    q_vec: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Same as search_index, but reuses a precomputed query vector to avoid
    embedding the query multiple times (e.g., for EN + AR passes).
    """
    scores, idxs = index.search(q_vec, top_k)

    out = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
        if idx == -1:
            continue
        it = items[idx]
        out.append(
            {
                "rank": rank,
                "score": float(score),
                "id": it.id,
                "meta": it.meta,
                "text": it.text,
            }
        )
    return out


def pick_base_group(indexes: Dict[str, Any], lang: str, topic: str, source_type: str) -> Tuple[Optional[faiss.IndexFlatIP], List[KBItem]]:
    """
    Picks the smallest available prebuilt group:
      1) lang+topic+source
      2) lang+topic
      3) lang
    Returns (index, items) or (None, []).
    """
    lang = lang or "en"
    topic = topic or "any"
    source_type = source_type or "any"

    if lang not in indexes or not indexes[lang].get("all"):
        return None, []

    if topic != "any" and topic in indexes[lang]["topic"]:
        tnode = indexes[lang]["topic"][topic]
        if source_type != "any" and source_type in tnode["source"]:
            return tnode["source"][source_type]
        if tnode.get("all"):
            return tnode["all"]

    # fallback to lang-only
    return indexes[lang]["all"]


def filter_items_by_tags(items: List[KBItem], wanted_tags: List[str]) -> List[KBItem]:
    if not wanted_tags:
        return items
    wanted = {t.strip() for t in wanted_tags if t.strip()}
    out = []
    for it in items:
        tags = set(get_alignment_tags(it.meta))
        if wanted.issubset(tags):
            out.append(it)
    return out


def retrieve(
    client: OpenAI,
    indexes: Dict[str, Any],
    query: str,
    emb_model: str,
    top_k: int,
    lang: str,
    topic: str,
    source_type: str,
    tags: List[str],
    target_audience: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # CROSS-LINGUAL LOGIC:
    # If lang is 'all' or 'any', we gather candidates from BOTH 'en' and 'ar' indexes.
    langs_to_search = ["en", "ar"] if (not lang or lang in ["all", "any"]) else [lang]

    # Embed the query ONCE and reuse for all language/index passes
    q_vec = embed_openai(client, [query], emb_model).astype("float32")

    all_candidates = []

    for l in langs_to_search:
        # We reuse the existing logic but force the language 'l'
        base_index, base_items = pick_base_group(indexes, l, topic, source_type)
        if base_index is None or not base_items:
            continue

        # Filter by tags if needed
        if tags:
            base_items = filter_items_by_tags(base_items, tags)

        # Filter by Target Audience if needed
        if target_audience and target_audience != "any":
            base_items = [
                it
                for it in base_items
                if str(it.meta.get("target_audience", "general")).lower()
                in [target_audience, "general", "both"]
            ]

        if not base_items:
            continue

        # If we filtered down items, we must build a temp index (or use the base if no filtering happened)
        # To be safe and simple: if we filtered (tags OR audience), build temp.
        # If we didn't filter, use base_index.

        use_index = base_index
        items_to_search = base_items

        needs_temp_index = bool(tags) or (bool(target_audience) and target_audience != "any")

        if needs_temp_index:
            temp = make_index_from_items(base_items)
            if temp:
                use_index = temp
            else:
                continue
        else:
            # We use the prebuilt base_index, but we must ensure base_items matches it 1:1.
            # pick_base_group returns the items that match the index.
            pass

        # Search this language/group
        results = search_index_with_vec(use_index, items_to_search, q_vec, top_k)
        all_candidates.extend(results)

    # Sort combined results by score (descending)
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by ID (just in case)
    seen = set()
    unique = []
    for c in all_candidates:
        if c["id"] not in seen:
            unique.append(c)
            seen.add(c["id"])
            
    return unique[:top_k]


# -----------------------------
# Context + generation
# -----------------------------
def build_context(retrieved: List[Dict[str, Any]], min_score: float, max_chars: int) -> str:
    kept = [r for r in retrieved if r["score"] >= min_score]

    # Light-touch prioritization by entry_kind so the LLM
    # sees summaries/plans/protocols slightly before raw cases.
    PRIORITY_MAP = {
        "protocol": 0,
        "taxonomy": 0,
        "curated_summary": 1,
        "curated_plan": 1,
        "principle": 2,
        "practice": 2,
        "curated_lessons": 2,
    }

    def kind_priority(meta: Dict[str, Any]) -> int:
        inner = meta.get("meta")
        kind = ""
        if isinstance(inner, dict):
            kind = str(inner.get("entry_kind", "")).lower()
        return PRIORITY_MAP.get(kind, 3)

    kept.sort(key=lambda r: (kind_priority(r["meta"]), -r["score"]))

    blocks = []
    for r in kept:
        meta_bits = []
        for k in ("source_type", "language", "topic"):
            if k in r["meta"]:
                meta_bits.append(f"{k}={r['meta'][k]}")
        # surface tags if present (nice for debugging)
        tags = get_alignment_tags(r["meta"])
        if tags:
            meta_bits.append(f"tags={','.join(tags[:8])}" + ("…" if len(tags) > 8 else ""))
        meta_str = (" | " + ", ".join(meta_bits)) if meta_bits else ""
        blocks.append(f"[{r['id']}] score={r['score']:.3f}{meta_str}\n{r['text']}")
    ctx = "\n\n---\n\n".join(blocks)
    return ctx[:max_chars]


def generate_answer(client: OpenAI, chat_model: str, user_query: str, context: str) -> str:
    system = (
        "You are MoslehAI, a culturally respectful marital guidance assistant.\n"
        "Use ONLY the provided context.\n"
        "If the context is not enough, say: "
        "\"I don't have enough info in my knowledge base to answer that confidently.\"\n"
        "Keep it practical, calm, and safe."
    )

    user = f"Context:\n{context if context.strip() else '(no context)'}\n\nUser query:\n{user_query}"

    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb_moslehai_v3_final.jsonl", help="Path to KB JSONL (default: kb_moslehai_v3_final.jsonl)")
    ap.add_argument("--emb_model", default="text-embedding-3-small", help="OpenAI embedding model")
    ap.add_argument("--chat_model", default="gpt-4o-mini", help="OpenAI chat model for optional generation")
    ap.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve")
    ap.add_argument("--min_score", type=float, default=0.25, help="Minimum cosine score to include in context")
    ap.add_argument("--max_context_chars", type=int, default=6000, help="Max chars fed into the LLM")
    ap.add_argument("--generate", action="store_true", help="Generate an answer using retrieved context")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY env var. Set it first.")

    client = OpenAI()

    print(f"Loading KB: {args.kb}")
    items = load_kb_jsonl(args.kb)
    print(f"Loaded {len(items)} KB items.")

    print(f"Embedding + building FAISS indexes with: {args.emb_model}")
    indexes = build_indexes(client, items, args.emb_model)

    en_count = len(indexes.get("en", {}).get("all", (None, []))[1]) if indexes.get("en", {}).get("all") else 0
    ar_count = len(indexes.get("ar", {}).get("all", (None, []))[1]) if indexes.get("ar", {}).get("all") else 0
    print(f"Indexes ready. EN items: {en_count} | AR items: {ar_count}\n")

    while True:
        raw_q = input("Query (or 'exit'): ").strip()
        if not raw_q or raw_q.lower() == "exit":
            break

        filters, q = parse_query_filters(raw_q)

        # If user did not specify lang, auto-detect from the query
        # (you can still override with [lang=en], [lang=ar], or [lang=all])
        lang = filters["lang"] or detect_language(q)
        
        topic = filters["topic"] or infer_topic(q)
        source_type = filters["source_type"] or "any"
        tags = filters["tags"] or []
        target_audience = filters.get("target_audience")

        # normalize
        lang = lang.lower()
        topic = topic.lower() if topic else "any"
        source_type = source_type.lower() if source_type else "any"
        if target_audience: target_audience = target_audience.lower()

        print(f"(Lang: {lang} | topic={topic} | source={source_type} | audience={target_audience} | tags={tags})")

        # SAFETY OVERRIDE: If topic is safety, lower the bar to ensuring protocols are always seen.
        context_min_score = args.min_score
        if topic == "safety":
            print(">> SAFETY MODE: Lowering min_score threshold to 0.20 to catch critical protocols.")
            context_min_score = 0.20

        retrieved = retrieve(
            client=client,
            indexes=indexes,
            query=q,
            emb_model=args.emb_model,
            top_k=args.top_k,
            lang=lang,
            topic=topic,
            source_type=source_type,
            tags=tags,
            target_audience=target_audience
        )

        print("\n=== RETRIEVED ===")
        if not retrieved:
            print("(no results)\n")
        else:
            for r in retrieved:
                preview = (r["text"][:260] + "…") if len(r["text"]) > 260 else r["text"]
                print(f"{r['rank']}) score={r['score']:.3f}  id={r['id']}  meta={r['meta']}")
                print(f"   {preview}\n")
        
        context = build_context(retrieved, context_min_score, args.max_context_chars)

        print("=== CONTEXT USED (after min_score filter) ===")
        print(context if context.strip() else "(nothing passed min_score)")
        print()

        if args.generate:
            print("=== GENERATED ANSWER ===")
            ans = generate_answer(client, args.chat_model, q, context)
            print(ans)
            print()

    print("Done.")


if __name__ == "__main__":
    main()
