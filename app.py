"""RAG Document Q&A — Streamlit App.

Upload a PDF, ask questions, get answers with page citations.
Uses local Ollama for generation, sentence-transformers for embeddings,
ChromaDB for vector storage, and cross-encoder for reranking.
"""

import streamlit as st

from rag.chunker import extract_text_from_pdf, chunk_pages
from rag.retriever import index_chunks, retrieve, get_collection
from rag.generator import generate_answer

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Sidebar: document upload ────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Parsing PDF..."):
                pages = extract_text_from_pdf(uploaded_file.read())

            with st.spinner("Chunking text..."):
                chunks = chunk_pages(pages, chunk_size=300, overlap=50)

            with st.spinner("Embedding & indexing..."):
                n = index_chunks(chunks)

            st.session_state["doc_ready"] = True
            st.session_state["doc_name"] = uploaded_file.name
            st.session_state["num_chunks"] = n
            st.session_state["num_pages"] = len(pages)
            st.session_state["messages"] = []
            st.success(f"Indexed **{n} chunks** from **{len(pages)} pages**")

    if st.session_state.get("doc_ready"):
        st.divider()
        st.caption(f"**Document:** {st.session_state['doc_name']}")
        st.caption(f"**Pages:** {st.session_state['num_pages']}")
        st.caption(f"**Chunks:** {st.session_state['num_chunks']}")

    st.divider()
    st.caption("Built with Ollama + ChromaDB + Streamlit")

# ── Main area ────────────────────────────────────────────────
st.title("📄 RAG Document Q&A")
st.markdown("Upload a PDF in the sidebar, then ask questions about it.")

if not st.session_state.get("doc_ready"):
    st.info("👈 Upload and process a PDF to get started.")
    st.stop()

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}** (relevance: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

# ── Chat input ───────────────────────────────────────────────
if query := st.chat_input("Ask a question about your document..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            sources = retrieve(query, top_k=10, rerank_top=5)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, sources)

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"**Page {src['page']}** (relevance: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
