import streamlit as st
import requests
import base64
import os

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Voice Chatbot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#css
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Source tag */
    .source-box {
        background: #21262d;
        border-left: 3px solid #7c6af7;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.78em;
        color: #8b949e;
        margin-top: 6px;
    }

    /* Status badge */
    .badge-green {
        display: inline-block;
        background: #238636;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75em;
    }
    .badge-red {
        display: inline-block;
        background: #da3633;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75em;
    }

    /* Headings */
    h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# Session State Init 
def init_state():
    defaults = {
        "messages": [],
        "index_built": False,
        "doc_names": [],
        "total_chunks": 0,
        "chunk_strategy": "overlap",
        "tts_enabled": True,
        "last_audio": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


def play_audio(audio_b64: str):
    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def ask_question(query: str):
    try:
        res = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "query": query,
                "enable_tts": st.session_state.tts_enabled
            },
            timeout=30
        )
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"Backend error {res.status_code}: {res.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is `uvicorn main:app` running?"}
    except Exception as e:
        return {"error": str(e)}


                       
with st.sidebar:
    st.markdown("## 🎙️ RAG Voice Chatbot")
    st.markdown("---")

    if st.session_state.index_built:
        st.markdown('<span class="badge-green">● Index Ready</span>', unsafe_allow_html=True)
        st.caption(f"📄 Docs: {', '.join(st.session_state.doc_names)}")
        st.caption(f"🧩 Chunks: {st.session_state.total_chunks}")
    else:
        st.markdown('<span class="badge-red">● No Index</span>', unsafe_allow_html=True)
        st.caption("Upload PDFs and build the index to start.")

    st.markdown("---")

    st.markdown("### 📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose up to 2 PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="These are the documents the chatbot will answer from."
    )

    if uploaded_files and len(uploaded_files) > 2:
        st.error("⚠️ Maximum 2 PDFs allowed.")
        uploaded_files = uploaded_files[:2]

    st.markdown("### ✂️ Chunking Strategy")
    strategy = st.radio(
        "Select method:",
        options=["overlap", "fixed", "semantic"],
        index=0,
        help="How to split PDF text into chunks for retrieval."
    )

    with st.expander("ℹ️ What do these mean?"):
        st.markdown("""
**Overlap** *(recommended)*  
Chunks share boundary words. Prevents losing info at edges.

**Fixed**  
Equal-sized word chunks. Fastest but may cut mid-sentence.

**Semantic**  
Splits where meaning changes. Best quality, slowest to build.
        """)

    if st.button("🔨 Build Index", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("Please upload at least 1 PDF.")
        else:
            with st.spinner("Reading PDFs and building FAISS index..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.read(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    res = requests.post(
                        f"{BACKEND_URL}/build-index",
                        files=files_payload,
                        data={"chunk_strategy": strategy},
                        timeout=120
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.index_built = True
                        st.session_state.doc_names = data.get("documents", [])
                        st.session_state.total_chunks = data.get("total_chunks", 0)
                        st.session_state.chunk_strategy = strategy
                        st.session_state.messages = []  # reset chat
                        st.success(f"✅ Index built! {data['total_chunks']} chunks from {len(data['documents'])} PDF(s).")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend not reachable. Run: `uvicorn main:app --reload`")

    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    st.session_state.tts_enabled = st.toggle(
        "🔊 Text-to-Speech",
        value=True,
        help="Auto-play the chatbot's response as audio."
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with FastAPI + GROQ + FAISS + Streamlit")


st.markdown("# 🎙️ RAG Voice Chatbot")
st.markdown("Ask questions from your uploaded PDFs — by **voice** or **text**.")
st.markdown("---")

st.markdown("### 🎤 Voice Input")

try:
    from audio_recorder_streamlit import audio_recorder

    col1, col2 = st.columns([1, 3])
    with col1:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#7c6af7",
            neutral_color="#484f58",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=2.5,
            sample_rate=16000
        )
    with col2:
        st.markdown("""
        <div style='padding-top:10px; color:#8b949e;'>
        🔴 Click mic to record<br>
        ⏹️ Pause 2.5s to stop<br>
        Audio is auto-transcribed and sent as a query
        </div>
        """, unsafe_allow_html=True)

    if audio_bytes and len(audio_bytes) > 2000:
        if st.session_state.last_audio != audio_bytes:
            st.session_state.last_audio = audio_bytes

            if not st.session_state.index_built:
                st.warning("⚠️ Build the index first using the sidebar.")
            else:
                with st.spinner("🎧 Transcribing your voice..."):
                    try:
                        stt_res = requests.post(
                            f"{BACKEND_URL}/transcribe",
                            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
                            timeout=30
                        )
                        if stt_res.status_code == 200:
                            voice_query = stt_res.json().get("text", "")
                            st.info(f"🗣️ **You said:** {voice_query}")

                            if voice_query and "error" not in voice_query.lower():
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": voice_query,
                                    "input_type": "voice"
                                })

                                with st.spinner("🔍 Searching documents..."):
                                    result = ask_question(voice_query)

                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": result.get("answer", ""),
                                        "sources": result.get("sources", []),
                                        "audio_b64": result.get("audio_b64")
                                    })
                                    st.rerun()
                        else:
                            st.error("Transcription failed.")
                    except Exception as e:
                        st.error(f"Voice error: {e}")

except ImportError:
    st.info("💡 Install `audio-recorder-streamlit` for voice input: `pip install audio-recorder-streamlit`")
    audio_bytes = None

st.markdown("---")

st.markdown("### 💬 Chat")

if not st.session_state.messages:
    if st.session_state.index_built:
        st.markdown(
            "<div style='text-align:center; color:#484f58; padding:40px;'>"
            "📄 Index ready! Ask your first question below."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center; color:#484f58; padding:40px;'>"
            "👈 Upload PDFs in the sidebar and click <b>Build Index</b> to get started."
            "</div>",
            unsafe_allow_html=True
        )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Show voice badge for voice inputs
        if msg.get("input_type") == "voice":
            st.markdown("🎤 *via voice*")

        st.write(msg["content"])

        # Show sources
        if msg.get("sources"):
            st.markdown(
                f'<div class="source-box">📄 Sources: {" | ".join(msg["sources"])}</div>',
                unsafe_allow_html=True
            )

        if msg.get("audio_b64"):
            audio_bytes_decoded = base64.b64decode(msg["audio_b64"])
            # Only autoplay the LATEST message
            autoplay = (i == len(st.session_state.messages) - 1)
            st.audio(audio_bytes_decoded, format="audio/mp3", autoplay=autoplay)

user_input = st.chat_input(
    "Type your question here..." if st.session_state.index_built else "Build index first..."
)

if user_input:
    if not st.session_state.index_built:
        st.warning("Please upload PDFs and build the index first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "input_type": "text"
        })

        with st.spinner("🔍 Searching documents and generating answer..."):
            result = ask_question(user_input)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.pop()  # remove the user msg if error
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
                "audio_b64": result.get("audio_b64")
            })

        st.rerun()