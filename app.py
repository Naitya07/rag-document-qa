"""DocQ — NotebookLM-style RAG Document Q&A.

Upload any document, image, audio, or video — ask questions, get answers with citations.
Multiple workspaces displayed as a card grid, like Google NotebookLM.
"""

import re
from datetime import datetime

import streamlit as st

from rag.chunker import extract_text, chunk_pages
from rag.retriever import index_chunks, retrieve, reset_collection
from rag.generator import generate_answer

SUPPORTED_TYPES = [
    "pdf", "docx", "pptx", "txt", "csv", "md", "html", "htm", "json",
    "png", "jpg", "jpeg", "webp",
    "mp3", "wav", "mp4", "mov", "webm", "m4a",
]

# Icons for workspaces (assigned based on dominant file type)
WS_ICONS = ["📄", "📊", "🎓", "🔬", "📚", "💡", "🗂️", "📝", "🧪", "🎯"]

st.set_page_config(page_title="DocQ", page_icon="🔮", layout="wide")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Top bar ── */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #222;
        margin-bottom: 2rem;
    }
    .topbar-logo {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #48C6EF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Notebook cards grid ── */
    .nb-card {
        background: linear-gradient(135deg, #1a1d25, #22252e);
        border: 1px solid #2a2d35;
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.2s, transform 0.2s;
        cursor: pointer;
    }
    .nb-card:hover {
        border-color: #6C63FF;
        transform: translateY(-2px);
    }
    .nb-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .nb-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #eee;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    .nb-meta {
        font-size: 0.78rem;
        color: #666;
        margin-top: auto;
    }

    .nb-card-create {
        background: #14161c;
        border: 2px dashed #333;
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: border-color 0.2s;
        cursor: pointer;
    }
    .nb-card-create:hover { border-color: #6C63FF; }
    .nb-plus {
        width: 56px; height: 56px;
        border-radius: 50%;
        background: #22252e;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        color: #6C63FF;
        margin-bottom: 0.8rem;
    }
    .nb-create-text {
        color: #888;
        font-size: 0.95rem;
    }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ddd;
        margin-bottom: 1rem;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #6C63FF; }
    .stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

    /* ── Source cards ── */
    .source-card {
        background: #1a1d23;
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .source-file { color: #6C63FF; font-weight: 600; font-size: 0.85rem; }
    .source-score { color: #48C6EF; font-size: 0.75rem; float: right; }
    .source-text { color: #999; font-size: 0.8rem; margin-top: 4px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #12141a; }
    .sidebar-header { font-size: 1.1rem; font-weight: 700; color: #6C63FF; margin-bottom: 0.5rem; }

    .format-grid { display: flex; flex-wrap: wrap; gap: 4px; margin: 4px 0; }
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

    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── State init ───────────────────────────────────────────────
def _safe_collection_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    if not safe or not safe[0].isalpha():
        safe = "ws_" + safe
    return safe[:63]


if "workspaces" not in st.session_state:
    st.session_state["workspaces"] = {}
if "active_ws" not in st.session_state:
    st.session_state["active_ws"] = None
if "view" not in st.session_state:
    st.session_state["view"] = "home"  # "home" or "chat"
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "creating" not in st.session_state:
    st.session_state["creating"] = False


def get_ws():
    name = st.session_state.get("active_ws")
    if name and name in st.session_state["workspaces"]:
        return st.session_state["workspaces"][name]
    return None


def get_ws_icon(index):
    return WS_ICONS[index % len(WS_ICONS)]


# ── Route: HOME view (card grid) ────────────────────────────
if st.session_state["view"] == "home":

    # Top bar
    st.markdown("""
    <div class="topbar">
        <div class="topbar-logo">DocQ</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your Workspaces</div>', unsafe_allow_html=True)

    ws_names = list(st.session_state["workspaces"].keys())
    # 4 columns: first is always "+ Create new", rest are workspaces
    cols_per_row = 4
    all_items = ["__CREATE__"] + ws_names
    rows = [all_items[i:i + cols_per_row] for i in range(0, len(all_items), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if j >= len(row):
                break
            item = row[j]
            with col:
                if item == "__CREATE__":
                    if st.button("➕ Create new workspace", key="create_new_btn", use_container_width=True):
                        st.session_state["creating"] = True
                        st.rerun()
                    st.markdown("""
                    <div style="text-align:center; color:#666; font-size:0.8rem; margin-top:4px;">
                        Upload files and start asking
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    ws_data = st.session_state["workspaces"][item]
                    icon = get_ws_icon(ws_names.index(item))
                    file_count = len(ws_data["doc_names"])
                    date_str = ws_data.get("created", "")
                    msg_count = len(ws_data["messages"])

                    if st.button(
                        f"{icon}  {item}",
                        key=f"open_{item}",
                        use_container_width=True,
                    ):
                        st.session_state["active_ws"] = item
                        st.session_state["view"] = "chat"
                        st.rerun()

                    st.markdown(f"""
                    <div style="color:#666; font-size:0.78rem; margin-top:2px;">
                        {date_str} · {file_count} source{'s' if file_count != 1 else ''} · {msg_count} message{'s' if msg_count != 1 else ''}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🗑️", key=f"del_{item}", help=f"Delete {item}"):
                        del st.session_state["workspaces"][item]
                        if st.session_state["active_ws"] == item:
                            st.session_state["active_ws"] = None
                        st.rerun()

    # ── Create new workspace dialog ──────────────────────────
    if st.session_state.get("creating"):
        st.markdown("")
        st.markdown("---")
        st.markdown('<div class="section-title">Create New Workspace</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            new_ws_name = st.text_input("Workspace name", placeholder="e.g. Lecture Notes")
        with c2:
            st.markdown("")

        uploaded_files = st.file_uploader(
            "Upload files for this workspace",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
        )

        col_create, col_cancel, _ = st.columns([1, 1, 3])
        with col_create:
            if uploaded_files and new_ws_name:
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
                        "created": datetime.now().strftime("%b %d, %Y"),
                    }
                    st.session_state["active_ws"] = new_ws_name
                    st.session_state["creating"] = False
                    st.session_state["uploader_key"] += 1
                    st.session_state["view"] = "chat"
                    st.rerun()
            else:
                st.button("Create Workspace", type="primary", use_container_width=True, disabled=True)

        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state["creating"] = False
                st.session_state["uploader_key"] += 1
                st.rerun()

        if uploaded_files:
            st.caption(f"**{len(uploaded_files)}** file(s) ready")
        if not new_ws_name and uploaded_files:
            st.warning("Give your workspace a name first")

    # ── Feature cards at bottom ──────────────────────────────
    if not ws_names and not st.session_state.get("creating"):
        st.markdown("")
        st.markdown("")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border:1px solid #333;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📄</div>
                <div style="font-weight:600;color:#fff;margin-bottom:0.3rem;">Documents</div>
                <div style="color:#888;font-size:0.85rem;">PDF, Word, PowerPoint, Markdown, HTML, CSV, JSON</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border:1px solid #333;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🖼️</div>
                <div style="font-weight:600;color:#fff;margin-bottom:0.3rem;">Images (OCR)</div>
                <div style="color:#888;font-size:0.85rem;">Extract text from screenshots, scanned docs, photos</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border:1px solid #333;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🎙️</div>
                <div style="font-weight:600;color:#fff;margin-bottom:0.3rem;">Audio & Video</div>
                <div style="color:#888;font-size:0.85rem;">Transcribe lectures, meetings, podcasts with Whisper</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#1e1e2e,#2a2a3e);border:1px solid #333;border-radius:12px;padding:1.5rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
                <div style="font-weight:600;color:#fff;margin-bottom:0.3rem;">Smart Search</div>
                <div style="color:#888;font-size:0.85rem;">Vector search + cross-encoder reranking for accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.stop()


# ── Route: CHAT view (active workspace) ─────────────────────
ws = get_ws()
if not ws:
    st.session_state["view"] = "home"
    st.rerun()

active_name = st.session_state["active_ws"]
collection_name = ws["collection"]

# ── Sidebar (only visible in chat view) ─────────────────────
with st.sidebar:
    if st.button("← Back to all workspaces", use_container_width=True):
        st.session_state["view"] = "home"
        st.rerun()

    st.markdown("")
    st.markdown(f'<div class="sidebar-header">{active_name}</div>', unsafe_allow_html=True)
    st.caption(f"Created {ws.get('created', '')}")

    st.divider()

    st.markdown(f"**Sources ({len(ws['doc_names'])})**")
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

# ── Stats bar ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(ws['doc_names'])}</div>
        <div class="stat-label">Sources</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{ws['num_pages']}</div>
        <div class="stat-label">Sections</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{ws['num_chunks']}</div>
        <div class="stat-label">Chunks Indexed</div>
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
                        <span class="source-score">Section {src['page']} · {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────────
if query := st.chat_input(f"Ask about \"{active_name}\"..."):
    ws["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching across all sources..."):
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
                        <span class="source-score">Section {src['page']} · {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

    ws["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
