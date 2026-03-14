"""RAG Document Q&A — Streamlit App.

Upload any document, image, audio, or video — ask questions, get answers with citations.
Supports multiple workspaces to work on different document sets simultaneously.
"""

import re
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

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

    [data-testid="stSidebar"] {
        background: #12141a;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.5rem;
    }

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

    .feature-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    .feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 600; color: #fff; margin-bottom: 0.3rem; }
    .feature-desc { color: #888; font-size: 0.85rem; }

    .ws-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin: 4px 0;
    }
    .ws-card-active {
        border-color: #6C63FF;
        background: linear-gradient(135deg, #1e1e3e, #2a2a4e);
    }
    .ws-name {
        font-weight: 600;
        color: #fff;
        font-size: 0.9rem;
    }
    .ws-meta {
        color: #888;
        font-size: 0.7rem;
    }

    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Workspace helpers ────────────────────────────────────────
def _safe_collection_name(name: str) -> str:
    """Convert workspace name to a valid ChromaDB collection name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    if not safe or not safe[0].isalpha():
        safe = "ws_" + safe
    return safe[:63]  # ChromaDB max 63 chars


if "workspaces" not in st.session_state:
    st.session_state["workspaces"] = {}  # {name: {doc_names, num_chunks, num_pages, messages, collection}}
if "active_ws" not in st.session_state:
    st.session_state["active_ws"] = None


def get_ws():
    """Get active workspace data or None."""
    name = st.session_state.get("active_ws")
    if name and name in st.session_state["workspaces"]:
        return st.session_state["workspaces"][name]
    return None


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    # ── Workspace selector ───────────────────────────────────
    st.markdown('<div class="sidebar-header">Workspaces</div>', unsafe_allow_html=True)

    ws_names = list(st.session_state["workspaces"].keys())

    if ws_names:
        for ws_name in ws_names:
            ws_data = st.session_state["workspaces"][ws_name]
            is_active = ws_name == st.session_state.get("active_ws")
            card_class = "ws-card ws-card-active" if is_active else "ws-card"
            file_count = len(ws_data["doc_names"])
            chunk_count = ws_data["num_chunks"]

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    f"{'▸ ' if is_active else ''}{ws_name}",
                    key=f"ws_switch_{ws_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_ws"] = ws_name
                    st.rerun()
            with col_del:
                if st.button("✕", key=f"ws_del_{ws_name}", help=f"Delete {ws_name}"):
                    del st.session_state["workspaces"][ws_name]
                    if st.session_state["active_ws"] == ws_name:
                        remaining = list(st.session_state["workspaces"].keys())
                        st.session_state["active_ws"] = remaining[0] if remaining else None
                    st.rerun()

            if is_active:
                st.caption(f"  {file_count} files • {chunk_count} chunks")

    st.divider()

    # ── Create new workspace ─────────────────────────────────
    st.markdown('<div class="sidebar-header">New Workspace</div>', unsafe_allow_html=True)

    new_ws_name = st.text_input("Workspace name", placeholder="e.g. Lecture Notes", label_visibility="collapsed")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and new_ws_name:
        st.caption(f"**{len(uploaded_files)}** file(s) selected")
        if st.button("Create Workspace", type="primary", use_container_width=True):
            collection_name = _safe_collection_name(new_ws_name)
            reset_collection(collection_name)
            total_chunks = 0
            total_sections = 0
            file_names = []

            progress = st.progress(0, text="Starting...")
            for i, uploaded_file in enumerate(uploaded_files):
                name = uploaded_file.name
                file_names.append(name)
                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing **{name}**...",
                )
                pages = extract_text(uploaded_file.read(), name)
                chunks = chunk_pages(pages, chunk_size=300, overlap=50)
                n = index_chunks(chunks, filename=name, collection_name=collection_name)
                total_chunks += n
                total_sections += len(pages)

            progress.progress(1.0, text="Done!")

            st.session_state["workspaces"][new_ws_name] = {
                "doc_names": file_names,
                "num_chunks": total_chunks,
                "num_pages": total_sections,
                "messages": [],
                "collection": collection_name,
            }
            st.session_state["active_ws"] = new_ws_name
            st.rerun()

    elif uploaded_files and not new_ws_name:
        st.warning("Give your workspace a name first")

    st.divider()

    # ── Active workspace files ───────────────────────────────
    ws = get_ws()
    if ws:
        st.markdown(f"**{len(ws['doc_names'])} files in \"{st.session_state['active_ws']}\"**")
        with st.expander("View all files", expanded=False):
            for name in ws["doc_names"]:
                ext = name.rsplit(".", 1)[-1].lower()
                if ext in ("png", "jpg", "jpeg", "webp"):
                    badge = "format-badge format-badge-img"
                elif ext in ("mp3", "wav", "mp4", "mov", "webm", "m4a"):
                    badge = "format-badge format-badge-media"
                else:
                    badge = "format-badge format-badge-doc"
                st.markdown(f'<span class="{badge}">{ext.upper()}</span> {name}', unsafe_allow_html=True)
        st.divider()

    # ── Supported formats ────────────────────────────────────
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
ws = get_ws()

if not ws:
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
            <div class="feature-title">Multi-Workspace</div>
            <div class="feature-desc">Work on 4-5 different document sets at the same time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    with st.expander("How it works — Architecture", expanded=False):
        st.code("""
    File Upload → Text Extraction / OCR / Transcription
                        ↓
                  Chunking (overlapping segments)
                        ↓
                  Embedding (sentence-transformers)
                        ↓
                  Vector Store (ChromaDB) ← one collection per workspace
                        ↓
    User Query → Query Embedding → Vector Search → Cross-Encoder Rerank
                                                        ↓
                                        Top-K Chunks + Query → LLM (Ollama)
                                                        ↓
                                              Answer with Source Citations
        """, language=None)

    st.info("👈 Name a workspace, upload files, and click **Create Workspace** to get started")
    st.stop()

# ── Active workspace view ────────────────────────────────────
active_name = st.session_state["active_ws"]
collection_name = ws["collection"]

# Stats bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state['workspaces'])}</div>
        <div class="stat-label">Workspaces</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(ws['doc_names'])}</div>
        <div class="stat-label">Files</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{ws['num_pages']}</div>
        <div class="stat-label">Sections</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{ws['num_chunks']}</div>
        <div class="stat-label">Chunks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ── Chat history ─────────────────────────────────────────────
for msg in ws["messages"]:
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
if query := st.chat_input(f"Ask about \"{active_name}\"..."):
    ws["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching across all documents..."):
            sources = retrieve(query, top_k=20, rerank_top=8, collection_name=collection_name)

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

    ws["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
