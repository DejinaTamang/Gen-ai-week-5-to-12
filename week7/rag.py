

import os
from pathlib import Path

import chromadb
from anthropic import Anthropic

from retriever import build_index, load_docs_folder, load_model, retrieve

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

LLM_MODEL = "claude-haiku-5-5" 
MIN_SIMILARITY = 0.30          
NO_ANSWER = "I could not find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered context passages provided.
Rules:
- Use only information found in the context passages.
- After each fact, cite the passage it came from, like [1] or [2].
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- Do not use outside knowledge."""


def make_llm_client() -> Anthropic:
    """Create the Anthropic client, with a clear error if the key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. In PowerShell, run: "
            '$env:ANTHROPIC_API_KEY="your-key-here"'
        )
    return Anthropic()


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Format the retrieved chunks as numbered context, followed by the question."""
    context = "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer_question(llm, embed_model, collection, question: str, top_k: int = 3) -> dict:
    """Retrieve context, then ask the LLM to answer from it."""
    retrieved = retrieve(embed_model, collection, question, top_k)
    relevant = [c for c in retrieved if c["similarity"] >= MIN_SIMILARITY]

    # Nothing relevant found: skip the LLM call entirely
    if not relevant:
        return {"answer": NO_ANSWER, "sources": [], "retrieved": retrieved}

    response = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, relevant)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return {"answer": text.strip(), "sources": relevant, "retrieved": retrieved}


TEST_QUESTIONS = [
    "How many days do I have to return a damaged item?",
    "When is customer support open?",
    "How much does express shipping cost?",
    "Do you ship internationally?",     # not in the documents
    "What is the capital of France?",   # completely unrelated
]


def main() -> None:
    llm = make_llm_client()
    embed_model = load_model()

    docs = load_docs_folder(DOCS_DIR)
    client = chromadb.EphemeralClient()
    collection = build_index(embed_model, client, docs)
    print(f"Indexed {len(docs)} documents ({collection.count()} chunks).\n")

    for question in TEST_QUESTIONS:
        result = answer_question(llm, embed_model, collection, question)
        print("=" * 70)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            print("Sources used:")
            for s in result["sources"]:
                print(f"   - {s['source']} (similarity {s['similarity']:.3f})")
        print()


if __name__ == "__main__":
    main()