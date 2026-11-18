"""
Shared helpers for Weeks 6 to 8.
Loads documents, splits them into chunks, stores embeddings in ChromaDB,
and retrieves the most relevant chunks for a question.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # same model as Week 5
COLLECTION_NAME = "docs_collection"


def load_model() -> SentenceTransformer:
    """Load the local embedding model (downloads once, then cached)."""
    return SentenceTransformer(MODEL_NAME)


def load_docs_folder(folder: Path) -> dict[str, str]:
    """Read every .txt file in a folder into a dict of {filename: text}."""
    docs = {}
    for path in sorted(folder.glob("*.txt")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    """
    Split text into chunks of roughly `chunk_size` words.
    Neighboring chunks share `overlap` words so an idea that spans
    a boundary is not lost.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_index(
    model: SentenceTransformer,
    client: chromadb.ClientAPI,
    docs: dict[str, str],
    chunk_size: int = 60,
    overlap: int = 15,
):
    """
    Chunk every document, embed the chunks, and store them in a fresh
    ChromaDB collection that uses cosine distance.
    """
    # Start fresh each time so old chunks don't linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for source, text in docs.items():
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            ids.append(f"{source}-{i}")
            texts.append(chunk)
            metadatas.append({"source": source})

    if texts:
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    return collection


def retrieve(
    model: SentenceTransformer,
    collection,
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Return the top_k most relevant chunks for a question.
    Each result has: text, source, and similarity (1 = identical meaning).
    """
    if not question.strip() or collection.count() == 0:
        return []

    query_vec = model.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_vec,
        n_results=min(top_k, collection.count()),
    )

    hits = []
    for doc, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # Chroma returns cosine distance; similarity = 1 - distance
        hits.append({
            "text": doc,
            "source": meta["source"],
            "similarity": 1 - distance,
        })
    return hits