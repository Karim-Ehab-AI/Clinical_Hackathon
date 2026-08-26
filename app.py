import os
import streamlit as st
import requests
from typing import Dict, Any, List

# Backend API Configuration
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:3000")
HEALTH_URL = f"{API_BASE_URL}/health"
INGEST_URL = f"{API_BASE_URL}/api/v1/ingestion/ingest-pdf"
GENERATE_URL = f"{API_BASE_URL}/api/v1/generation/generate"

# Page Config
st.set_page_config(
    page_title="Your Diabetes AI Assistant",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS applying EVA AI Recruiter Design System Theme
# Warm dark canvas: #15151a, Cards: #242429, Accent: #38bdf8, Borders: #34343c, Foreground: #f5f5f5
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Neutral Canvas */
    .stApp {
        background-color: #15151a;
        color: #f5f5f5;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1f;
        border-right: 1px solid #34343c;
    }
    
    /* Main Greeting Container */
    .welcome-container {
        text-align: center;
        padding: 4rem 1rem 2rem 1rem;
    }
    .welcome-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5f5f5;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #f5f5f5 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-sub {
        font-size: 1.1rem;
        color: #a0a0a8;
    }

    /* Status Badge */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .pill-ready {
        background-color: rgba(98, 179, 122, 0.15);
        color: #62b37a;
        border: 1px solid rgba(98, 179, 122, 0.4);
    }
    .pill-not-ready {
        background-color: rgba(85, 85, 95, 0.2);
        color: #a0a0a8;
        border: 1px solid #34343c;
    }

    /* Refusal Card */
    .refusal-card {
        background-color: #242429;
        border: 1px solid #e5675a;
        border-left: 4px solid #e5675a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #f5f5f5;
        margin-top: 0.5rem;
    }
    
    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #242429;
        border: 1px solid #34343c;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Popover button styling */
    div[data-testid="stPopover"] > button {
        background-color: #2b2b32 !important;
        color: #38bdf8 !important;
        border: 1px solid #34343c !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.2rem 0.6rem !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background-color: #32323a !important;
        border-color: #38bdf8 !important;
    }

    /* Hide Streamlit Default Components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def check_system_readiness() -> bool:
    """Check whether system backend services are operational."""
    try:
        res = requests.get(HEALTH_URL, timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def render_citation_popover(cit: Dict[str, Any], idx: int):
    """Render popover showing ONLY non-None metadata fields and the original source chunk text."""
    chunk_id = cit.get("chunk_id")
    page = cit.get("pdf_page")
    section = cit.get("section")
    rec_id = cit.get("recommendation_id")
    source_text = cit.get("source_text")

    # Build list of non-None, non-N/A metadata lines
    meta_lines = []
    if chunk_id:
        meta_lines.append(f"**Chunk ID:** `{chunk_id}`")
    if page is not None and str(page).strip() and str(page).upper() != "N/A":
        meta_lines.append(f"📄 **PDF Page:** {page}")
    if section and str(section).strip() and str(section).upper() != "N/A":
        meta_lines.append(f"📑 **Section:** {section}")
    if rec_id and str(rec_id).strip() and str(rec_id).upper() != "N/A":
        meta_lines.append(f"🎯 **Recommendation ID:** {rec_id}")

    label = f"📄 Source Citation #{idx + 1}"
    if page:
        label += f" (Page {page})"

    with st.popover(label):
        for line in meta_lines:
            st.markdown(line)

        if source_text:
            st.markdown("---")
            st.markdown("**Original Source Text:**")
            st.caption(f'"{source_text}"')


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 👨‍⚕️ Your Diabetes AI Assistant")
    st.caption("Clinical Decision Support")

    # System Readiness Status Badge
    is_ready = check_system_readiness()
    if is_ready:
        st.markdown(
            '<div class="status-pill pill-ready"><span>🟢</span> System Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill pill-not-ready"><span>🔘</span> System Not Ready</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Upload PDF Section (Clean UI - No technical jargon)
    uploaded_file = st.file_uploader(
        "Upload PDF Guideline",
        type=["pdf"],
        help="Upload new clinical guidelines to expand knowledge base.",
    )

    if uploaded_file is not None:
        if st.button("📤 Process & Upload", use_container_width=True, type="primary"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(INGEST_URL, files=files, timeout=120)
                    if res.status_code == 200:
                        st.success("✅ File uploaded & indexed successfully!")
                    else:
                        st.error(f"Upload failed: {res.status_code}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT AREA ---

# Dynamic Greeting Header: Disappears once a message exists!
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-sub">Ask any clinical question regarding Type 2 Diabetes management.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggested Preset Query Pills
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Who should be offered isCGM monitoring?", use_container_width=True):
            st.session_state.preset_query = "Who should be offered isCGM monitoring in adults with type 2 diabetes?"
            st.rerun()
    with col2:
        if st.button("💡 Blood glucose monitoring recommendations?", use_container_width=True):
            st.session_state.preset_query = "What is recommended regarding blood glucose monitoring for adults with type 2 diabetes?"
            st.rerun()

# Display Chat History (2-Sided: Human vs Doctor Avatar)
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "👨‍⚕️"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            res_data = message.get("data", {})
            result = res_data.get("result", {})
            is_sufficient = result.get("is_knowledge_sufficient", False)
            answer = result.get("answer")
            refusal_reason = result.get("refusal_reason")
            citations = result.get("citations", [])

            if is_sufficient and answer:
                st.markdown(answer)

                if citations:
                    st.markdown("<br>", unsafe_allow_html=True)
                    cols = st.columns(len(citations))
                    for idx, cit in enumerate(citations):
                        with cols[idx]:
                            render_citation_popover(cit, idx)
            else:
                refusal_msg = refusal_reason or "عذراً، لا يمكن إجابة هذا السؤال بناءً على المعلومات المتاحة."
                st.markdown(
                    f"""
                    <div class="refusal-card">
                        <strong>🛡️ Clinical Refusal:</strong><br>
                        {refusal_msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# User Chat Input Bar
query_input = st.chat_input("Ask a question about Type 2 Diabetes...")

# Handle Preset Click
if "preset_query" in st.session_state and st.session_state.preset_query:
    query_input = st.session_state.preset_query
    del st.session_state.preset_query

# Handle User Submission
if query_input:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": query_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query_input)

    # 2. Call Generation Endpoint & Render Doctor Response
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        with st.spinner("Analyzing clinical context..."):
            try:
                res = requests.post(GENERATE_URL, json={"query": query_input}, timeout=60)
                if res.status_code == 200:
                    res_data = res.json()
                    result = res_data.get("result", {})

                    is_sufficient = result.get("is_knowledge_sufficient", False)
                    answer = result.get("answer")
                    refusal_reason = result.get("refusal_reason")
                    citations = result.get("citations", [])

                    if is_sufficient and answer:
                        st.markdown(answer)

                        if citations:
                            st.markdown("<br>", unsafe_allow_html=True)
                            cols = st.columns(len(citations))
                            for idx, cit in enumerate(citations):
                                with cols[idx]:
                                    render_citation_popover(cit, idx)
                    else:
                        refusal_msg = refusal_reason or "عذراً، لا يمكن إجابة هذا السؤال بناءً على المعلومات المتاحة."
                        st.markdown(
                            f"""
                            <div class="refusal-card">
                                <strong>🛡️ Clinical Refusal:</strong><br>
                                {refusal_msg}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Save Assistant Response to Session History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer if is_sufficient else refusal_msg,
                        "data": res_data,
                    })
                else:
                    st.error(f"Generation error ({res.status_code}): {res.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")
