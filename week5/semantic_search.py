"""
Week 5: Tiny Semantic Search with Embeddings

Given a text query, find the most semantically similar sentences
from a small dataset (sentences.txt).

Usage:
    python semantic_search.py                          # interactive mode
    python semantic_search.py "how do I get my money back"   # single query
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# A small, fast, free model that runs locally. It produces 384-dimensional vectors.
MODEL_NAME = "all-MiniLM-L6-v2"

# Below this similarity score, we consider nothing a good match.
SIMILARITY_THRESHOLD = 0.30

DATA_FILE = Path(__file__).parent / "sentences.txt"


def load_sentences(path: Path) -> list[str]:
    """Read the dataset: one sentence per line, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    if not sentences:
        raise ValueError(f"No sentences found in {path}")
    return sentences


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D array of vectors.
    Shape: (number_of_texts, embedding_dimension)
    """
    return model.encode(texts, convert_to_numpy=True)


def cosine_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between one query vector and many document vectors.

    cosine(a, b) = (a · b) / (|a| * |b|)

    Returns an array of scores, one per document, each between -1 and 1.
    """
    # Length of each document vector and of the query vector
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    query_norm = np.linalg.norm(query_vec)

    # Guard against division by zero for empty vectors
    doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
    query_norm = query_norm if query_norm != 0 else 1e-10

    # Dot product of the query with every document, divided by both lengths
    dot_products = doc_matrix @ query_vec
    return dot_products / (doc_norms * query_norm)


def semantic_search(
    query: str,
    model: SentenceTransformer,
    sentences: list[str],
    sentence_vecs: np.ndarray,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Embed the query, score it against every sentence, and return the top_k
    most similar sentences as (sentence, score) pairs, highest score first.
    """
    query_vec = embed_texts(model, [query])[0]
    scores = cosine_scores(query_vec, sentence_vecs)

    # argsort sorts ascending, so reverse it to get highest scores first
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [(sentences[i], float(scores[i])) for i in ranked_indices]


def print_results(query: str, results: list[tuple[str, float]]) -> None:
    """Display results, or a message if nothing is similar enough."""
    print(f'\nQuery: "{query}"')

    best_score = results[0][1] if results else 0.0
    if best_score < SIMILARITY_THRESHOLD:
        print("  No close match found in the dataset.")
        return

    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"  {rank}. [{score:.3f}] {sentence}")


def main() -> None:
    # Step 1: Load the dataset
    print("Loading sentences...")
    sentences = load_sentences(DATA_FILE)
    print(f"Loaded {len(sentences)} sentences.")

    # Step 2: Load the embedding model (downloads on first run, then cached)
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # Step 3: Precompute embeddings for all sentences (done once)
    print("Generating embeddings for the dataset...")
    sentence_vecs = embed_texts(model, sentences)
    print(f"Embeddings shape: {sentence_vecs.shape}")

    # Step 4: Handle a single query from the command line, or enter interactive mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        if not query:
            print("Error: the query cannot be empty.")
            return
        results = semantic_search(query, model, sentences, sentence_vecs)
        print_results(query, results)
        return

    print("\nType a question and press Enter. Type 'quit' to exit.")
    while True:
        try:
            query = input("\nSearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:
            print("Please enter a non-empty query.")
            continue

        results = semantic_search(query, model, sentences, sentence_vecs)
        print_results(query, results)


if __name__ == "__main__":
    main()