"""Cortex — AI-powered document intelligence.

Upload any document, image, audio, or video — ask questions, get answers with citations.
Multiple notebooks displayed as an interactive card grid.
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

WS_ICONS = ["🧠", "📚", "🔬", "🎓", "💡", "🗂️", "🧪", "🎯", "📊", "🚀"]
WS_GRADIENTS = [
    "linear-gradient(135deg, #667eea, #764ba2)",
    "linear-gradient(135deg, #f093fb, #f5576c)",
    "linear-gradient(135deg, #4facfe, #00f2fe)",
    "linear-gradient(135deg, #43e97b, #38f9d7)",
    "linear-gradient(135deg, #fa709a, #fee140)",
    "linear-gradient(135deg, #a18cd1, #fbc2eb)",
    "linear-gradient(135deg, #fccb90, #d57eeb)",
    "linear-gradient(135deg, #e0c3fc, #8ec5fc)",
    "linear-gradient(135deg, #f5576c, #ff9a9e)",
    "linear-gradient(135deg, #667eea, #764ba2)",
]

st.set_page_config(page_title="Cortex", page_icon="🧠", layout="wide")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }

    /* ── Animated background ── */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(108,99,255,0.1); }
        50% { box-shadow: 0 0 40px rgba(108,99,255,0.3); }
    }

    /* ── Hero ── */
    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        animation: fadeInUp 0.8s ease-out;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.75rem;
        color: #6C63FF;
        margin-bottom: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 30%, #f093fb 60%, #6C63FF 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    .hero-subtitle {
        color: #777;
        font-size: 1.15rem;
        max-width: 600px;
        margin: 0 auto 0.5rem auto;
        line-height: 1.5;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
    }
    .hero-stat {
        text-align: center;
    }
    .hero-stat-num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .hero-stat-label {
        font-size: 0.7rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Notebook cards ── */
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 1.2rem;
        margin-top: 1rem;
    }

    .nb-card {
        background: #16181f;
        border: 1px solid #2a2d35;
        border-radius: 16px;
        padding: 0;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
    }
    .nb-card:hover {
        border-color: #6C63FF;
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108,99,255,0.15);
    }
    .nb-card-banner {
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
    }
    .nb-card-body {
        padding: 1rem 1.2rem 1.2rem 1.2rem;
    }
    .nb-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #eee;
        margin-bottom: 0.4rem;
    }
    .nb-card-meta {
        font-size: 0.75rem;
        color: #666;
    }
    .nb-card-tags {
        display: flex;
        gap: 4px;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }
    .nb-tag {
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.65rem;
        color: #8b85ff;
    }

    /* Create card */
    .create-card {
        background: #12141a;
        border: 2px dashed #2a2d35;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 180px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeInUp 0.4s ease-out;
    }
    .create-card:hover {
        border-color: #6C63FF;
        background: #16181f;
    }
    .create-plus {
        width: 60px; height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6C63FF, #48C6EF);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        color: white;
        margin-bottom: 0.8rem;
        animation: float 3s ease-in-out infinite;
        box-shadow: 0 8px 30px rgba(108,99,255,0.3);
    }
    .create-text {
        color: #888;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .create-subtext {
        color: #555;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }

    /* ── Feature strip ── */
    .feature-strip {
        display: flex;
        gap: 1rem;
        margin-top: 2.5rem;
        animation: fadeInUp 1s ease-out;
    }
    .feature-pill {
        flex: 1;
        background: linear-gradient(135deg, #16181f, #1e2028);
        border: 1px solid #2a2d35;
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .feature-pill:hover {
        border-color: #6C63FF;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .feature-pill-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .feature-pill-title { font-weight: 600; color: #ddd; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .feature-pill-desc { color: #666; font-size: 0.72rem; line-height: 1.4; }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ccc;
    }
    .section-count {
        background: rgba(108,99,255,0.15);
        border-radius: 10px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #6C63FF;
        font-weight: 600;
    }

    /* ── Stat cards (chat view) ── */
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
    .format-badge { background: #2a2a3e; border: 1px solid #333; border-radius: 6px; padding: 2px 8px; font-size: 0.65rem; color: #aaa; }
    .format-badge-doc { border-color: #6C63FF; color: #6C63FF; }
    .format-badge-img { border-color: #48C6EF; color: #48C6EF; }
    .format-badge-media { border-color: #FF6B6B; color: #FF6B6B; }

    .stChatMessage { border-radius: 12px; }

    /* ── Create form ── */
    .create-form {
        background: #16181f;
        border: 1px solid #2a2d35;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
        animation: fadeInUp 0.4s ease-out;
    }
</style>
""", unsafe_allow_html=True)


# ── State ────────────────────────────────────────────────────
def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    if not safe or not safe[0].isalpha():
        safe = "ws_" + safe
    return safe[:63]

for key, default in [
    ("workspaces", {}), ("active_ws", None),
    ("view", "home"), ("uploader_key", 0), ("creating", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_ws():
    name = st.session_state.get("active_ws")
    if name and name in st.session_state["workspaces"]:
        return st.session_state["workspaces"][name]
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HOME VIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state["view"] == "home":

    ws_dict = st.session_state["workspaces"]
    ws_names = list(ws_dict.keys())
    total_sources = sum(len(w["doc_names"]) for w in ws_dict.values())
    total_chats = sum(len(w["messages"]) // 2 for w in ws_dict.values())

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-badge">Powered by Local AI — 100% Private</div>
        <div class="hero-title">Cortex</div>
        <div class="hero-subtitle">
            Your AI research assistant. Upload documents, images, audio, or video —
            then have a conversation with your knowledge.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-num">{len(ws_names)}</div>
                <div class="hero-stat-label">Notebooks</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">{total_sources}</div>
                <div class="hero-stat-label">Sources</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">{total_chats}</div>
                <div class="hero-stat-label">Conversations</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-num">18</div>
                <div class="hero-stat-label">File Types</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section header ───────────────────────────────────────
    if ws_names:
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">Recent notebooks</div>
            <div class="section-count">{len(ws_names)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Get started</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Card grid ────────────────────────────────────────────
    cols_per_row = 4
    # First slot is always the create card
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
                    if st.button("🆕  Create new notebook", key="create_btn", use_container_width=True, type="primary"):
                        st.session_state["creating"] = True
                        st.rerun()
                    st.markdown("""
                    <div style="text-align:center; color:#555; font-size:0.75rem; margin-top:4px;">
                        PDF, DOCX, images, audio, video & more
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    ws_data = ws_dict[item]
                    idx = ws_names.index(item)
                    icon = WS_ICONS[idx % len(WS_ICONS)]
                    gradient = WS_GRADIENTS[idx % len(WS_GRADIENTS)]
                    file_count = len(ws_data["doc_names"])
                    date_str = ws_data.get("created", "")
                    msg_count = len(ws_data["messages"]) // 2

                    # Get top file extensions for tags
                    exts = set()
                    for f in ws_data["doc_names"][:5]:
                        exts.add(f.rsplit(".", 1)[-1].upper())
                    tags_html = "".join(f'<span class="nb-tag">{e}</span>' for e in sorted(exts)[:4])

                    if st.button(f"{icon}  {item}", key=f"open_{item}", use_container_width=True):
                        st.session_state["active_ws"] = item
                        st.session_state["view"] = "chat"
                        st.rerun()

                    st.markdown(f"""
                    <div style="margin-top:-4px;">
                        <div class="nb-card-meta">{date_str} · {file_count} sources · {msg_count} chats</div>
                        <div class="nb-card-tags">{tags_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🗑️", key=f"del_{item}", help=f"Delete {item}"):
                        del st.session_state["workspaces"][item]
                        if st.session_state["active_ws"] == item:
                            st.session_state["active_ws"] = None
                        st.rerun()

    # ── Create form ──────────────────────────────────────────
    if st.session_state.get("creating"):
        st.markdown("---")
        st.markdown("### 📓 Create a new notebook")
        st.caption("Give it a name, upload your sources, and start asking questions.")

        c1, c2 = st.columns([1, 2])
        with c1:
            new_ws_name = st.text_input("Notebook name", placeholder="e.g. COMP-4400 Midterm Prep")
        with c2:
            st.markdown("")

        uploaded_files = st.file_uploader(
            "Upload sources",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
            help="Drop any combination of documents, images, audio, or video",
        )

        if uploaded_files:
            st.caption(f"**{len(uploaded_files)}** file(s) ready to process")

        col_create, col_cancel, _ = st.columns([1, 1, 3])
        with col_create:
            can_create = bool(uploaded_files and new_ws_name)
            if st.button(
                "Create Notebook",
                type="primary",
                use_container_width=True,
                disabled=not can_create,
            ):
                collection_name = _safe_name(new_ws_name)
                reset_collection(collection_name)
                total_chunks = 0
                total_sections = 0
                file_names = []

                progress = st.progress(0, text="Starting...")
                for i, uf in enumerate(uploaded_files):
                    name = uf.name
                    file_names.append(name)
                    progress.progress(i / len(uploaded_files), text=f"Processing **{name}**...")
                    pages = extract_text(uf.read(), name)
                    chunks = chunk_pages(pages, chunk_size=300, overlap=50)
                    n = index_chunks(chunks, filename=name, collection_name=collection_name)
                    total_chunks += n
                    total_sections += len(pages)

                progress.progress(1.0, text="All done!")

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

        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state["creating"] = False
                st.session_state["uploader_key"] += 1
                st.rerun()

    # ── Feature strip ────────────────────────────────────────
    if not st.session_state.get("creating"):
        st.markdown("""
        <div class="feature-strip">
            <div class="feature-pill">
                <div class="feature-pill-icon">📄</div>
                <div class="feature-pill-title">18 File Formats</div>
                <div class="feature-pill-desc">PDF, DOCX, PPTX, MD, CSV, HTML, JSON, TXT, images, audio, video</div>
            </div>
            <div class="feature-pill">
                <div class="feature-pill-icon">👁️</div>
                <div class="feature-pill-title">Image OCR</div>
                <div class="feature-pill-desc">Read text from screenshots, scanned pages, whiteboard photos</div>
            </div>
            <div class="feature-pill">
                <div class="feature-pill-icon">🎤</div>
                <div class="feature-pill-title">Audio & Video</div>
                <div class="feature-pill-desc">Transcribe lectures, podcasts, and meetings with Whisper AI</div>
            </div>
            <div class="feature-pill">
                <div class="feature-pill-icon">🔒</div>
                <div class="feature-pill-title">100% Local</div>
                <div class="feature-pill-desc">Everything runs on your machine. No data leaves your laptop.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHAT VIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ws = get_ws()
if not ws:
    st.session_state["view"] = "home"
    st.rerun()

active_name = st.session_state["active_ws"]
collection_name = ws["collection"]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    if st.button("← All notebooks", use_container_width=True):
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
    st.caption("COMP-4400 · University of Windsor")

# ── Stats ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(ws["doc_names"])}</div><div class="stat-label">Sources</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{ws["num_pages"]}</div><div class="stat-label">Sections</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{ws["num_chunks"]}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Chat ─────────────────────────────────────────────────────
for msg in ws["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} references)"):
                for src in msg["sources"]:
                    source_name = src.get("source", "Unknown")
                    score = src["score"]
                    preview = src["text"][:250] + "..." if len(src["text"]) > 250 else src["text"]
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-file">{source_name}</span>
                        <span class="source-score">Section {src['page']} · {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

if query := st.chat_input(f"Ask about \"{active_name}\"..."):
    ws["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching sources..."):
            sources = retrieve(query, top_k=20, rerank_top=8, collection_name=collection_name)
        with st.spinner("💭 Thinking..."):
            answer = generate_answer(query, sources)

        st.markdown(answer)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} references)"):
                for src in sources:
                    source_name = src.get("source", "Unknown")
                    score = src["score"]
                    preview = src["text"][:250] + "..." if len(src["text"]) > 250 else src["text"]
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-file">{source_name}</span>
                        <span class="source-score">Section {src['page']} · {score:.2f}</span>
                        <div class="source-text">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

    ws["messages"].append({"role": "assistant", "content": answer, "sources": sources})
