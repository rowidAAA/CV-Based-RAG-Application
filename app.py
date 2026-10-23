"""Streamlit UI for a section-aware CV RAG assistant."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from rag import (
    SECTION_NAMES,
    answer_question,
    embed_metadata_headers,
    get_qdrant_client,
    index_cv,
    infer_query_sections,
    load_metadata_registry,
    read_cv_with_links,
    select_candidate_headers,
)


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@st.cache_resource(show_spinner=False)
def cached_qdrant_client(path: str):
    return get_qdrant_client(path)

st.set_page_config(page_title="CV RAG Assistant", page_icon="📄", layout="centered")
st.title("CV RAG Assistant")
st.caption("Ask questions about a CV and get answers grounded in its contents.")

with st.sidebar:
    st.header("CV and retrieval")
    uploaded_cv = st.file_uploader("Upload a PDF or text CV", type=["pdf", "txt"])
    selected_sections = st.multiselect(
        "Limit search to sections (optional)",
        SECTION_NAMES,
        help="Leave empty to infer relevant sections from your question. If no section is inferred, all sections are searched.",
    )
    header_threshold = st.slider(
        "Section header similarity threshold",
        min_value=0.0,
        max_value=0.8,
        value=float(os.getenv("HEADER_SIMILARITY_THRESHOLD", "0.18")),
        step=0.01,
        help="Only chunk headers whose embedding similarity to your question meets this threshold are searched.",
    )
    st.caption("CV text and vectors are stored locally in the Qdrant database folder.")
    st.divider()
    st.header("OpenAI")
    api_key = st.text_input(
        "API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Set OPENAI_API_KEY in the project .env file or enter it here.",
    )
    llm_model = st.text_input("Chat model", value=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    embedding_model = st.text_input(
        "Embedding model", value=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

if uploaded_cv is None:
    st.info("Upload your CV to build a searchable knowledge base.")
    st.markdown(
        "Try questions about **education**, **technical skills**, **work experience**, "
        "**projects**, **certifications**, or **achievements**."
    )
    st.stop()

try:
    cv_bytes = uploaded_cv.getvalue()
    cv_text, cv_links = read_cv_with_links(uploaded_cv.name, cv_bytes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not api_key:
    st.warning("Add an OpenAI API key to create embeddings and index this CV.")
    st.stop()

cv_id = hashlib.sha256(cv_bytes).hexdigest()[:20]
model_id = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:8]
collection_name = f"cv3_{cv_id}_{model_id}"
qdrant_path = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_storage"))
database_path = str((ROOT / qdrant_path).resolve()) if not Path(qdrant_path).is_absolute() else qdrant_path

try:
    client = cached_qdrant_client(database_path)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    with st.spinner("Checking the CV index..."):
        vector_store, chunk_count, newly_indexed = index_cv(
            cv_text,
            uploaded_cv.name,
            cv_id,
            links=cv_links,
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
    if newly_indexed or st.session_state.get("metadata_collection") != collection_name:
        metadata_registry = load_metadata_registry(client, collection_name)
        if not metadata_registry:
            raise ValueError("No chunk metadata was found in this Qdrant collection.")
        header_vectors = embed_metadata_headers(metadata_registry, embeddings)
        st.session_state.metadata_collection = collection_name
        st.session_state.metadata_registry = metadata_registry
        st.session_state.header_vectors = header_vectors
    else:
        metadata_registry = st.session_state.metadata_registry
        header_vectors = st.session_state.header_vectors
except Exception as exc:
    detail = str(exc).strip().replace(api_key, "[redacted]")[:500]
    st.error(f"Could not index the CV ({exc.__class__.__name__}): {detail}")
    st.stop()

status = "indexed now" if newly_indexed else "loaded from local Qdrant storage"
st.success(f"CV {status}: {chunk_count} chunks in collection `{collection_name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("active_cv") != collection_name:
    st.session_state.active_cv = collection_name
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved CV passages"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['section'].title()} · {source['file']} · chunk {source['chunk']}**"
                    )
                    st.markdown(f"> {source['text']}")

question = st.chat_input("Ask something about the CV")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    inferred_sections = selected_sections or infer_query_sections(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the selected CV sections and preparing an answer..."):
            try:
                scored_headers = select_candidate_headers(
                    question,
                    metadata_registry,
                    header_vectors,
                    embeddings,
                    threshold=header_threshold,
                )
                allowed_sections = set(selected_sections) if selected_sections else None
                candidate_headers = [
                    header
                    for header, _score in scored_headers
                    if allowed_sections is None
                    or any(
                        str(item.get("section_header_text")) == header
                        and item.get("section") in allowed_sections
                        for item in metadata_registry
                    )
                ]
                response, matches = answer_question(
                    vector_store,
                    question,
                    api_key=api_key,
                    model=llm_model,
                    sections=selected_sections or None,
                    candidate_headers=candidate_headers,
                )
                sources = [
                    {
                        "section": document.metadata.get("section", "unknown"),
                        "file": document.metadata.get("source_file", uploaded_cv.name),
                        "chunk": document.metadata.get("chunk_index", "?"),
                        "text": document.page_content,
                        "score": float(score),
                    }
                    for document, score in matches
                ]
                st.markdown(response)
                if inferred_sections:
                    st.caption("Likely sections: " + ", ".join(inferred_sections))
                else:
                    st.caption("Likely sections: none inferred; header similarity decides candidates")
                st.caption(
                    f"Metadata prefilter (cosine similarity ≥ {header_threshold:.2f}): "
                    + (
                        ", ".join(
                            f"{header} ({score:.2f})"
                            for header, score in scored_headers
                            if header in candidate_headers
                        )
                        or "no section headers passed; ask about a section shown in the CV or lower the threshold"
                    )
                )
                if sources:
                    with st.expander("Retrieved CV passages"):
                        for source in sources:
                            st.markdown(
                                f"**{source['section'].title()} · {source['file']} · chunk {source['chunk']}**"
                            )
                            st.markdown(f"> {source['text']}")
            except Exception as exc:
                detail = str(exc).strip().replace(api_key, "[redacted]")[:500]
                response = (
                    "I couldn't generate an answer. Check your OpenAI settings and try again.\n\n"
                    f"Details: {exc.__class__.__name__}: {detail or 'No additional details provided.'}"
                )
                sources = []
                st.error(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
