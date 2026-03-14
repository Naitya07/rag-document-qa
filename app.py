"""RAG Document Q&A — Streamlit App.

Upload any document, image, audio, or video — ask questions, get answers with citations.
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
st.set_page_config(page_title="DocQ — Ask Your Documents", page_icon="🔮", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 50%, #6F86D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* File pills in sidebar */
    .file-pill {
        background: #2a2a3e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 4px 10px;
        margin: 2px 0;
        font-size: 0.75rem;
        color: #ccc;
        display: inline-block;
    }

    /* Source cards */
    .source-card {
        background: #1a1d23;
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .source-file {
        color: #6C63FF;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-score {
        color: #48C6EF;
        font-size: 0.75rem;
        float: right;
    }
    .source-text {
        color: #999;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #12141a;
    }

    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.5rem;
    }

    /* Format badges */
    .format-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin: 4px 0;
    }
    .format-badge {
        background: #2a2a3e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.65rem;
        color: #aaa;
    }
    .format-badge-doc { border-color: #6C63FF; color: #6C63FF; }
    .format-badge-img { border-color: #48C6EF; color: #48C6EF; }
    .format-badge-media { border-color: #FF6B6B; color: #FF6B6B; }

    /* Landing cards */
    .feature-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        color: #888;
        font-size: 0.85rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"**{len(uploaded_files)}** file(s) selected")
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
                    text=f"Processing **{name}**...",
                )
                pages = extract_text(uploaded_file.read(), name)
                chunks = chunk_pages(pages, chunk_size=300, overlap=50)
                n = index_chunks(chunks, filename=name)
                total_chunks += n
                total_sections += len(pages)

            progress.progress(1.0, text="All files processed!")

            st.session_state["doc_ready"] = True
            st.session_state["doc_names"] = file_names
            st.session_state["num_chunks"] = total_chunks
            st.session_state["num_pages"] = total_sections
            st.session_state["messages"] = []

    if st.session_state.get("doc_ready"):
        st.divider()
        names = st.session_state["doc_names"]
        st.markdown(f"**{len(names)} files loaded**")
        with st.expander(f"View all files", expanded=False):
            for name in names:
                ext = name.rsplit(".", 1)[-1].lower()
                if ext in ("png", "jpg", "jpeg", "webp"):
                    badge = "format-badge format-badge-img"
                elif ext in ("mp3", "wav", "mp4", "mov", "webm", "m4a"):
                    badge = "format-badge format-badge-media"
                else:
                    badge = "format-badge format-badge-doc"
                st.markdown(f'<span class="{badge}">{ext.upper()}</span> {name}', unsafe_allow_html=True)

    st.divider()

    st.markdown("**Supported Formats**")
    st.markdown("""
    <div class="format-grid">
        <span class="format-badge format-badge-doc">PDF</span>
        <span class="format-badge format-badge-doc">DOCX</span>
        <span class="format-badge format-badge-doc">PPTX</span>
        <span class="format-badge format-badge-doc">TXT</span>
        <span class="format-badge format-badge-doc">MD</span>
        <span class="format-badge format-badge-doc">CSV</span>
        <span class="format-badge format-badge-doc">HTML</span>
        <span class="format-badge format-badge-doc">JSON</span>
        <span class="format-badge format-badge-img">PNG</span>
        <span class="format-badge format-badge-img">JPG</span>
        <span class="format-badge format-badge-img">WEBP</span>
        <span class="format-badge format-badge-media">MP3</span>
        <span class="format-badge format-badge-media">MP4</span>
        <span class="format-badge format-badge-media">WAV</span>
        <span class="format-badge format-badge-media">MOV</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Built for COMP-4400 — University of Windsor")

# ── Main area ────────────────────────────────────────────────
if not st.session_state.get("doc_ready"):
    # Landing page
    st.markdown('<div class="hero-title">DocQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Ask questions about any document, image, or video — powered by local AI</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Documents</div>
            <div class="feature-desc">PDF, Word, PowerPoint, Markdown, HTML, CSV, JSON</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🖼️</div>
            <div class="feature-title">Images (OCR)</div>
            <div class="feature-desc">Extract text from screenshots, scanned docs, photos</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎙️</div>
            <div class="feature-title">Audio & Video</div>
            <div class="feature-desc">Transcribe lectures, meetings, podcasts with Whisper</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Smart Search</div>
            <div class="feature-desc">Vector search + cross-encoder reranking for accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Architecture diagram
    with st.expander("How it works — Architecture", expanded=False):
        st.code("""
    File Upload → Text Extraction / OCR / Transcription
                        ↓
                  Chunking (overlapping segments)
                        ↓
                  Embedding (sentence-transformers)
                        ↓
                  Vector Store (ChromaDB)
                        ↓
    User Query → Query Embedding → Vector Search → Cross-Encoder Rerank
                                                        ↓
                                        Top-K Chunks + Query → LLM (Ollama)
                                                        ↓
                                              Answer with Source Citations
        """, language=None)

    st.info("👈 Upload files in the sidebar to get started")
    st.stop()

# ── Stats bar ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state['doc_names'])}</div>
        <div class="stat-label">Files Loaded</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state['num_pages']}</div>
        <div class="stat-label">Sections</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state['num_chunks']}</div>
        <div class="stat-label">Chunks Indexed</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} references)"):
                for src in msg["sources"]:
                    source_name = src.get('source', 'Unknown')
                    score = src['score']
                    preview = src["text"][:250] + "..." if len(src["text"]) > 250 else src["text"]
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-file">{source_name}</span>
                        <span class="source-score">Section {src['page']} • Score: {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────────
if query := st.chat_input("Ask anything about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching across all documents..."):
            sources = retrieve(query, top_k=20, rerank_top=8)

        with st.spinner("💭 Generating answer..."):
            answer = generate_answer(query, sources)

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)} references)"):
                for src in sources:
                    source_name = src.get('source', 'Unknown')
                    score = src['score']
                    preview = src["text"][:250] + "..." if len(src["text"]) > 250 else src["text"]
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-file">{source_name}</span>
                        <span class="source-score">Section {src['page']} • Score: {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
