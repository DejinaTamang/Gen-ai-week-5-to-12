"""
Week 6: Load documents into a vector database and query it.
Also compares the results with a brute-force search (Week 5 style).
"""

from pathlib import Path

import chromadb
import numpy as np

from retriever import (
    build_index,
    chunk_text,
    load_docs_folder,
    load_model,
    retrieve,
)

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

SAMPLE_QUERIES = [
    "How do I get my money back?",
    "How long does delivery take?",
    "I can't log in to my account",
    "When is support open?",
]


def brute_force_search(model, all_chunks, chunk_vecs, question, top_k=3):
    """Week 5 style: compare the question to every chunk in memory."""
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = chunk_vecs @ q_vec          # normalized vectors, so dot = cosine
    best = np.argsort(scores)[::-1][:top_k]
    return [(all_chunks[i]["text"], all_chunks[i]["source"], float(scores[i]))
            for i in best]


def main() -> None:
    # Step 1: Load documents
    docs = load_docs_folder(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No .txt files found in {DOCS_DIR}")
    print(f"Loaded {len(docs)} documents.")

    # Step 2: Load the embedding model
    model = load_model()

    # Step 3: Store chunks and embeddings in ChromaDB (persisted to disk)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = build_index(model, client, docs)
    print(f"Stored {collection.count()} chunks in the vector database.")

    # Step 4: Prepare the brute-force version for comparison
    all_chunks = []
    for source, text in docs.items():
        for chunk in chunk_text(text):
            all_chunks.append({"text": chunk, "source": source})
    chunk_vecs = model.encode(
        [c["text"] for c in all_chunks], normalize_embeddings=True
    )

    # Step 5: Run sample queries through both approaches
    for query in SAMPLE_QUERIES:
        print("\n" + "=" * 70)
        print(f'Query: "{query}"')

        print("\n  Vector database results:")
        for rank, hit in enumerate(retrieve(model, collection, query), 1):
            print(f"    {rank}. [{hit['similarity']:.3f}] ({hit['source']}) "
                  f"{hit['text'][:90]}...")

        print("\n  Brute-force results (Week 5 approach):")
        for rank, (text, source, score) in enumerate(
            brute_force_search(model, all_chunks, chunk_vecs, query), 1
        ):
            print(f"    {rank}. [{score:.3f}] ({source}) {text[:90]}...")


if __name__ == "__main__":
    main()