# Document Q&A (Simple RAG App)

Upload your own .txt or .md files and ask questions about them. The app finds the
most relevant passages with embeddings, then asks Claude to answer using only those passages.

## What it does

1. Splits each uploaded document into overlapping chunks.
2. Embeds the chunks with `all-MiniLM-L6-v2` and stores them in ChromaDB.
3. For each question, retrieves the most similar chunks.
4. Sends the chunks and the question to Claude, which answers with citations.
5. Shows the sources used so you can check the answer.

## Error handling

- No files uploaded, or empty files: the app shows a warning.
- No relevant content found: the app says so and skips the language model call.
- Missing `ANTHROPIC_API_KEY`: the app shows setup instructions.
- API failures: the app shows the error instead of crashing.

## Setup

From the project folder, with your virtual environment active:

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

- Windows PowerShell: `$env:ANTHROPIC_API_KEY="your-key-here"`
- macOS / Linux: `export ANTHROPIC_API_KEY="your-key-here"`

## Run

```bash
streamlit run app.py
```

Open the local address Streamlit prints (usually http://localhost:8501).

## Usage

1. Upload one or more .txt or .md files in the sidebar.
2. Click "Index documents".
3. Type a question and click "Ask".

## Files

| File               | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `app.py`           | The Streamlit interface                      |
| `rag.py`           | Retrieval plus answer generation with Claude |
| `retriever.py`     | Chunking, embedding, and vector search       |
| `requirements.txt` | Python packages                              |

## Testing notes

Test with a few real questions and record whether each answer is correct, grounded in
the sources, and handled well when the answer isn't in the documents.
