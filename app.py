"""RAG Document Q&A — Streamlit App.

Upload any document, image, audio, or video — ask questions, get answers with citations.
Uses local Ollama for generation, sentence-transformers for embeddings,
ChromaDB for vector storage, and cross-encoder for reranking.
"""

import streamlit as st

from rag.chunker import extract_text, chunk_pages
from rag.retriever import index_chunks, retrieve, get_collection
from rag.generator import generate_answer

SUPPORTED_TYPES = [
    "pdf", "docx", "pptx", "txt", "csv", "md", "html", "htm", "json",
    "png", "jpg", "jpeg", "webp",
    "mp3", "wav", "mp4", "mov", "webm", "m4a",
]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Sidebar: document upload ────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        help="Documents, images, audio, video — we handle it all",
    )

    if uploaded_file:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Parsing document..."):
                pages = extract_text(uploaded_file.read(), uploaded_file.name)

            with st.spinner("Chunking text..."):
                chunks = chunk_pages(pages, chunk_size=300, overlap=50)

            with st.spinner("Embedding & indexing..."):
                n = index_chunks(chunks)

            st.session_state["doc_ready"] = True
            st.session_state["doc_name"] = uploaded_file.name
            st.session_state["num_chunks"] = n
            st.session_state["num_pages"] = len(pages)
            st.session_state["messages"] = []
            st.success(f"Indexed **{n} chunks** from **{len(pages)} sections**")

    if st.session_state.get("doc_ready"):
        st.divider()
        st.caption(f"**Document:** {st.session_state['doc_name']}")
        st.caption(f"**Sections:** {st.session_state['num_pages']}")
        st.caption(f"**Chunks:** {st.session_state['num_chunks']}")

    st.divider()
    st.markdown(
        "**Supported:**  \n"
        "Docs: PDF, DOCX, PPTX, TXT, MD, CSV, HTML, JSON  \n"
        "Images: PNG, JPG, WEBP (OCR)  \n"
        "Media: MP3, WAV, MP4, MOV (transcription)"
    )
    st.caption("Built with Ollama + ChromaDB + Whisper + Streamlit")

# ── Main area ────────────────────────────────────────────────
st.title("📄 RAG Document Q&A")
st.markdown("Upload a document in the sidebar, then ask questions about it.")

if not st.session_state.get("doc_ready"):
    st.info("👈 Upload and process a document to get started.")
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
                    st.markdown(f"**Section {src['page']}** (relevance: {src['score']:.2f})")
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
                    st.markdown(f"**Section {src['page']}** (relevance: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
