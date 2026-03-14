"""RAG Document Q&A — Streamlit App.

Upload any document, image, audio, or video — ask questions, get answers with citations.
Uses local Ollama for generation, sentence-transformers for embeddings,
ChromaDB for vector storage, and cross-encoder for reranking.
"""

import streamlit as st

from rag.chunker import extract_text, chunk_pages
from rag.retriever import index_chunks, retrieve, reset_collection
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
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Upload one or many — documents, images, audio, video",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        if st.button("Process All Files", type="primary", use_container_width=True):
            reset_collection()
            total_chunks = 0
            total_sections = 0
            file_names = []

            progress = st.progress(0, text="Starting...")
            for i, uploaded_file in enumerate(uploaded_files):
                name = uploaded_file.name
                file_names.append(name)
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing {name}...",
                )

                pages = extract_text(uploaded_file.read(), name)
                chunks = chunk_pages(pages, chunk_size=300, overlap=50)
                n = index_chunks(chunks, filename=name)
                total_chunks += n
                total_sections += len(pages)

            progress.progress(1.0, text="Done!")

            st.session_state["doc_ready"] = True
            st.session_state["doc_names"] = file_names
            st.session_state["num_chunks"] = total_chunks
            st.session_state["num_pages"] = total_sections
            st.session_state["messages"] = []
            st.success(f"Indexed **{total_chunks} chunks** from **{len(file_names)} file(s)**")

    if st.session_state.get("doc_ready"):
        st.divider()
        st.caption(f"**Files:** {len(st.session_state['doc_names'])}")
        for name in st.session_state["doc_names"]:
            st.caption(f"  - {name}")
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
st.markdown("Upload files in the sidebar, then ask questions about them.")

if not st.session_state.get("doc_ready"):
    st.info("👈 Upload and process files to get started.")
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
                    label = f"**{src['source']}** — Section {src['page']}" if src.get("source") else f"**Section {src['page']}**"
                    st.markdown(f"{label} (relevance: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

# ── Chat input ───────────────────────────────────────────────
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            sources = retrieve(query, top_k=20, rerank_top=8)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, sources)

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    label = f"**{src['source']}** — Section {src['page']}" if src.get("source") else f"**Section {src['page']}**"
                    st.markdown(f"{label} (relevance: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
