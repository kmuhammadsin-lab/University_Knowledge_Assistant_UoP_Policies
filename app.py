import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# University Knowledge Assistant — University of Peshawar
# Uses precomputed document vectors. Original PDFs are not needed.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DIR = PROJECT_DIR / "vector_store"
INDEX_PATH = VECTOR_DIR / "index.faiss"
METADATA_PATH = VECTOR_DIR / "chunks.jsonl"
CONFIG_PATH = VECTOR_DIR / "config.json"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


st.set_page_config(
    page_title="University Knowledge Assistant | UoP",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 University Knowledge Assistant")
st.caption("University of Peshawar — policies, rules, and department guidance")
st.info(
    "This assistant answers from the indexed text chunks available in its knowledge base. "
    "It is a navigation aid, not a replacement for current official regulations or written decisions."
)


@st.cache_resource(show_spinner="Loading the saved FAISS database and embedding model...")
def load_knowledge_base():
    """Load precomputed FAISS vectors and aligned JSONL metadata."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {METADATA_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    metadata: list[dict[str, Any]] = []
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                metadata.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {METADATA_PATH.name}, line {line_number}: {exc}"
                ) from exc

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS/metadata mismatch: "
            f"{index.ntotal} vectors but {len(metadata)} metadata records. "
            "The index and chunks.jsonl must come from the same ingestion run."
        )

    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

    model_name = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    model = SentenceTransformer(model_name)

    if model.get_sentence_embedding_dimension() != index.d:
        raise ValueError(
            "Embedding dimension mismatch. The configured embedding model does not "
            "match the saved FAISS index. Use the same model used during ingestion."
        )

    return index, metadata, model, config


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from environment variables first, then Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
        return str(value) if value else default
    except Exception:
        return default


def retrieve(question: str, index, metadata, embedding_model, top_k: int):
    """Embed the query and search normalized vectors using inner product/cosine similarity."""
    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    scores, positions = index.search(query_vector, min(top_k, index.ntotal))
    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        record = metadata[int(position)].copy()
        record["_similarity"] = float(score)
        results.append(record)

    return results


def format_source(record: dict[str, Any], number: int) -> str:
    title = record.get("document_title") or record.get("file_name") or "Unknown document"
    page = record.get("page_number", "Unknown")
    chunk_id = record.get("chunk_id", "Unknown")
    similarity = record.get("_similarity")
    score_text = f"{similarity:.3f}" if similarity is not None else "N/A"
    return (
        f"[{number}] **{title}** — page {page}  \n"
        f"Chunk ID: `{chunk_id}` · Similarity: `{score_text}`"
    )


def make_context(results: list[dict[str, Any]]) -> str:
    sections = []
    for i, record in enumerate(results, start=1):
        title = record.get("document_title") or record.get("file_name") or "Unknown document"
        page = record.get("page_number", "Unknown")
        chunk_id = record.get("chunk_id", "Unknown")
        text = record.get("text") or record.get("embedding_text") or ""
        sections.append(
            f"SOURCE [{i}]\n"
            f"Document: {title}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Text:\n{text}"
        )
    return "\n\n---\n\n".join(sections)


def answer_with_groq(question: str, context: str, api_key: str, model_name: str) -> str:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the University of Peshawar Knowledge Assistant. "
                    "Answer using only the supplied retrieved context. Do not invent "
                    "university policies, fees, deadlines, attendance percentages, penalties, "
                    "contacts, or exceptions. If the context does not establish the answer, "
                    "say that you could not confirm it from the available indexed documents "
                    "and direct the student to verify with the relevant official university "
                    "office or current notification. Distinguish general guidance from official "
                    "rules. Be concise and practical. Cite supporting sources inline using "
                    "the exact source labels [1], [2], etc., which correspond to the supplied "
                    "context. Mention the document title and page number when available."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    "Answer the question and cite the relevant source labels."
                ),
            },
        ],
        temperature=0.1,
        max_completion_tokens=1200,
    )

    return completion.choices[0].message.content or "The model returned an empty response."


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)
    groq_model = st.text_input("Groq model", value=DEFAULT_GROQ_MODEL)
    st.caption("The embedding model is loaded from vector_store/config.json.")

    st.divider()
    st.markdown("**Source traceability**")
    st.write("Answers display document title, page number, chunk ID, and similarity score.")
    st.caption("Similarity is a retrieval score, not a guarantee that a source is authoritative or current.")


try:
    index, metadata, embedding_model, config = load_knowledge_base()
except Exception as exc:
    st.error("Could not load the saved knowledge base.")
    st.code(str(exc))
    st.markdown(
        "Check that `vector_store/index.faiss`, `vector_store/chunks.jsonl`, and "
        "`vector_store/config.json` are present and belong to the same ingestion run."
    )
    st.stop()

with st.sidebar:
    st.success(f"Knowledge base loaded: {index.ntotal:,} chunks")
    st.caption(f"Embedding model: {config.get('embedding_model', DEFAULT_EMBEDDING_MODEL)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(source)

question = st.chat_input("Ask about admissions, registration, exams, attendance, fees, or university procedures...")

if question:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud → App settings → Secrets, "
            "or set it as an environment variable for local development."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy knowledge base and preparing an answer..."):
            try:
                results = retrieve(question, index, metadata, embedding_model, top_k)
                if not results:
                    response = (
                        "I could not find relevant text in the available indexed documents. "
                        "Please check the current official University of Peshawar source or "
                        "contact the relevant department/office."
                    )
                    source_lines = []
                else:
                    context = make_context(results)
                    response = answer_with_groq(question, context, api_key, groq_model)
                    source_lines = [
                        format_source(record, i)
                        for i, record in enumerate(results, start=1)
                    ]

                st.markdown(response)
                if source_lines:
                    with st.expander("View retrieved sources"):
                        for source in source_lines:
                            st.markdown(source)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": source_lines,
                    }
                )
            except Exception as exc:
                st.error("An error occurred while retrieving documents or calling Groq.")
                st.exception(exc)

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.divider()
st.caption(
    "Important: This assistant is a retrieval-based guide, not an official University decision. "
    "Verify current policies, deadlines, and department-specific requirements with official sources."
)
