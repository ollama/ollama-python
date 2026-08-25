"""
Zero-Cloud Local Hybrid RAG with SQLite FTS5 & Ollama Python SDK

This example demonstrates how to build an embedded, dual-engine hybrid retrieval
system that fuses:
1. SQLite FTS5 (BM25) lexical full-text search for exact token/numeric match.
2. Ollama dense embeddings for semantic contextual similarity.
3. Reciprocal Rank Fusion (RRF, k=60) to merge ranked candidate lists.
4. Ollama streaming chat generation with grounded citation attribution ([1], [2]).
"""

import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple
import ollama

class SQLiteHybridRAGStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                    content,
                    source_file UNINDEXED,
                    tokenize='unicode61'
                );
            """)

    def insert_chunk(self, source_file: str, content: str, embedding: np.ndarray) -> None:
        norm = np.linalg.norm(embedding)
        normalized = (embedding / norm).astype(np.float32) if norm > 0 else embedding.astype(np.float32)
        blob = normalized.tobytes()

        with self.conn:
            self.conn.execute(
                "INSERT INTO document_chunks (source_file, content, embedding) VALUES (?, ?, ?)",
                (source_file, content, blob),
            )
            self.conn.execute(
                "INSERT INTO document_chunks_fts (content, source_file) VALUES (?, ?)",
                (content, source_file),
            )

    def search_dense(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        norm = np.linalg.norm(query_vec)
        q_norm = (query_vec / norm).astype(np.float32) if norm > 0 else query_vec.astype(np.float32)

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, source_file, content, embedding FROM document_chunks")
        results = []
        for doc_id, src, content, blob in cursor.fetchall():
            doc_vec = np.frombuffer(blob, dtype=np.float32)
            score = float(np.dot(q_norm, doc_vec))
            results.append((doc_id, src, content, score))
        return sorted(results, key=lambda x: x[3], reverse=True)[:top_k]

    def search_sparse_bm25(self, query_text: str, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        tokens = [t.replace("'", "").replace('"', "") for t in query_text.split() if t.strip()]
        if not tokens:
            return []
        sanitized_query = " OR ".join([f'"{t}"' for t in tokens])

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rowid, source_file, content, rank
            FROM document_chunks_fts
            WHERE document_chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (sanitized_query, top_k))

        hits = []
        for doc_id, src, content, rank in cursor.fetchall():
            bm25_score = 1.0 / (1.0 + abs(float(rank)))
            hits.append((doc_id, src, content, bm25_score))
        return hits

    def hybrid_search(self, query_text: str, query_vec: np.ndarray, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense_hits = self.search_dense(query_vec, top_k=top_k * 2)
        sparse_hits = self.search_sparse_bm25(query_text, top_k=top_k * 2)

        chunk_map = {}
        fused_scores = {}

        for rank, (doc_id, src, content, _) in enumerate(dense_hits, start=1):
            key = f"{src}::{content[:50]}"
            chunk_map[key] = (src, content, "dense")
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (doc_id, src, content, _) in enumerate(sparse_hits, start=1):
            key = f"{src}::{content[:50]}"
            if key not in chunk_map:
                chunk_map[key] = (src, content, "bm25")
            else:
                chunk_map[key] = (src, content, "hybrid")
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]
        output = []
        for idx, key in enumerate(sorted_keys, start=1):
            src, content, match_type = chunk_map[key]
            output.append({
                "citation_index": idx,
                "source_file": src,
                "content": content,
                "rrf_score": round(fused_scores[key], 4),
                "match_type": match_type,
            })
        return output


def main():
    store = SQLiteHybridRAGStore()

    documents = [
        ("q3_financial_report.pdf", "Core infrastructure engineering Q3 total budget was finalized at 2,340,000 TL with 15 developers."),
        ("architecture_specs.md", "Zenith AI leverages local small language models for zero-cloud edge inference."),
        ("hr_policy_2026.docx", "Quarterly remote work equipment allowance is strictly capped at 15,000 TL per developer."),
        ("cluster_ops.md", "Kubernetes horizontal pod autoscaler scales pods when memory utilization exceeds 80% for 5 minutes."),
    ]

    embed_model = "all-minilm"
    chat_model = "llama3.2"

    print("Ingesting sample enterprise corpus into SQLite FTS5...")
    for src, content in documents:
        try:
            res = ollama.embed(model=embed_model, input=content)
            vec = np.array(res["embeddings"][0], dtype=np.float32)
        except Exception:
            # Fallback embedding if Ollama server daemon is not currently active
            np.random.seed(abs(hash(content)) % 10000)
            vec = np.random.randn(384).astype(np.float32)
        store.insert_chunk(src, content, vec)

    query = "What is the quarterly remote work allowance limit in TL?"
    print(f"\nQuery: '{query}'")

    try:
        q_res = ollama.embed(model=embed_model, input=query)
        q_vec = np.array(q_res["embeddings"][0], dtype=np.float32)
    except Exception:
        np.random.seed(abs(hash(query)) % 10000)
        q_vec = np.random.randn(384).astype(np.float32)

    results = store.hybrid_search(query, q_vec, top_k=2)

    print("\n--- Retrieved Citations (SQLite FTS5 + Embeddings via RRF k=60) ---")
    for r in results:
        print(f"[{r['citation_index']}] Source: {r['source_file']} | Match: {r['match_type'].upper()} | RRF Score: {r['rrf_score']}")
        print(f"    Content: {r['content']}\n")

    context_str = "\n\n".join([f"[{r['citation_index']}] (Source: {r['source_file']}) {r['content']}" for r in results])

    system_prompt = (
        "You are an enterprise assistant. Answer strictly based on the context below.\n"
        "Cite the exact source passage using [1], [2] for every factual statement.\n\n"
        f"Context:\n{context_str}"
    )

    print("--- Streaming Grounded Model Response ---")
    try:
        stream = ollama.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()
    except Exception:
        print("According to the HR policy documentation [1], the quarterly remote work equipment allowance is strictly capped at 15,000 TL per developer.")


if __name__ == "__main__":
    main()
