
"""
Week 8: A simple Streamlit app for asking questions about your own documents.

Run with: streamlit run app.py
"""

import chromadb
import streamlit as st
from groq import APIError

from rag import NO_ANSWER, answer_question, make_llm_client
from retriever import build_index, load_model


st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A")
st.caption("Upload .txt or .md files, then ask questions about them.")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Load the embedding model once and reuse it across reruns."""
    return load_model()


# ---------- Sidebar: upload and index documents ----------

with st.sidebar:
    st.header("1. Upload documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary"):

        if not uploaded_files:
            st.warning("Upload at least one .txt or .md file first.")

        else:
            docs = {}

            for f in uploaded_files:
                text = f.getvalue().decode(
                    "utf-8", errors="ignore"
                ).strip()

                if text:
                    docs[f.name] = text
                else:
                    st.warning(f"'{f.name}' is empty and was skipped.")

            if not docs:
                st.error("No readable text found in the uploaded files.")

            else:
                try:
                    with st.spinner("Splitting, embedding, and storing..."):

                        client = chromadb.EphemeralClient()

                        collection = build_index(
                            get_embedding_model(),
                            client,
                            docs
                        )

                    st.session_state["collection"] = collection
                    st.session_state["doc_names"] = list(docs)

                    st.success(
                        f"Indexed {len(docs)} file(s), "
                        f"{collection.count()} chunks."
                    )

                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if "doc_names" in st.session_state:
        st.markdown("**Currently indexed:**")

        for name in st.session_state["doc_names"]:
            st.write(f"- {name}")


# ---------- Main area: ask questions ----------

st.header("2. Ask a question")

question = st.text_input(
    "Your question",
    placeholder="e.g. How long do refunds take?"
)


if st.button("Ask"):

    if "collection" not in st.session_state:
        st.warning("Index some documents in the sidebar first.")

    elif not question.strip():
        st.warning("Please type a question.")

    else:
        try:
            llm = make_llm_client()

            with st.spinner(
                "Searching documents and generating an answer..."
            ):

                result = answer_question(
                    llm,
                    get_embedding_model(),
                    st.session_state["collection"],
                    question.strip(),
                )

            if result["answer"] == NO_ANSWER or not result["sources"]:
                st.info(
                    "No relevant content was found in your documents "
                    "for this question. Try rephrasing it or uploading "
                    "a document that covers this topic."
                )

            st.subheader("Answer")
            st.write(result["answer"])

            if result["sources"]:

                with st.expander("Sources used"):

                    for s in result["sources"]:

                        st.markdown(
                            f"**{s['source']}** "
                            f"(similarity {s['similarity']:.2f})"
                        )

                        st.caption(s["text"])

        except RuntimeError as e:
            # Raised when GROQ_API_KEY is missing
            st.error(str(e))

        except APIError as e:
            st.error(
                f"The Groq language model request failed: {e}"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
