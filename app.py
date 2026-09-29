import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import cleanup_audio, process_input

load_dotenv()

st.set_page_config(
    page_title="Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
    --ink: #172033;
    --muted: #5d6b82;
    --paper: #f4f6fa;
    --card: #ffffff;
    --line: #dde3ee;
    --accent: #2f5bea;
    --accent-soft: #e8eefe;
}

html, body, [class*="css"], .stMarkdown, .stTextInput input, .stChatInput textarea {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}
.stApp { background: var(--paper); }
.block-container { padding-top: 2.2rem; max-width: 1100px; }

h1, h2, h3, .hero-title {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--ink);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink);
}
section[data-testid="stSidebar"] * { color: #e7ecf7 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #24304a !important;
    border-color: #35446a !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #24304a;
    border: 1px dashed #4a5b85;
}
section[data-testid="stSidebar"] button {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] button:hover { background: #4a71f0 !important; }

/* hero */
.hero {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 6px solid var(--accent);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.hero-title { font-size: 2rem; font-weight: 700; margin: 0 0 .3rem 0; line-height: 1.15; }
.hero-sub { color: var(--muted); margin: 0; }

/* empty state */
.empty {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--muted);
}
.empty h2 { margin-bottom: .4rem; }

/* stats */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: .9rem 1.1rem;
}
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.6rem;
}

/* tabs */
button[data-baseweb="tab"] { font-weight: 500; font-size: 1rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* content panels */
.panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    line-height: 1.65;
}
.transcript {
    max-height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-size: .95rem;
}

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
st.session_state.setdefault("result", None)
st.session_state.setdefault("messages", [])


# ---------------------------------------------------------------- helpers
def as_text(value) -> str:
    """Extractors may return a string or a list; render either as markdown."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def panel(value, empty_msg: str) -> None:
    text = as_text(value).strip()
    if not text:
        st.info(empty_msg)
        return
    with st.container(border=True):
        st.markdown(text)


def build_report(r: dict) -> str:
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{as_text(r['summary'])}\n\n"
        f"## Action items\n{as_text(r['action_items'])}\n\n"
        f"## Key decisions\n{as_text(r['key_decisions'])}\n\n"
        f"## Open questions\n{as_text(r['open_questions'])}\n\n"
        f"## Transcript\n{r['transcript']}\n"
    )


def analyze(source: str, language: str) -> dict:
    language = language.strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("language must be 'english' or 'hinglish'")
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    if language == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("SARVAM_API_KEY is required when language is 'hinglish'.")
    with st.status("Working on your recording…", expanded=True) as status:
        st.write("Preparing audio")
        chunks = process_input(source)

        st.write("Transcribing")
        try:
            transcript = transcribe_all(chunks, language)
        finally:
            cleanup_audio(chunks, source)
        if not transcript.strip():
            status.update(label="No speech found", state="error")
            raise RuntimeError(
                "Transcription returned no text. Check that the recording has "
                "audible speech and that your transcription setup is working."
            )

        st.write("Writing title and summary")
        title = generate_title(transcript)
        summary = summarize(transcript)

        st.write("Finding action items, decisions and questions")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        st.write("Indexing the transcript for chat")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Done", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "chunks": len(chunks) if hasattr(chunks, "__len__") else None,
    }


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🎙️ Meeting Assistant")
    st.caption("Turn a recording into notes, then ask it questions.")

    mode = st.radio("Where is the recording?", ["YouTube link", "Upload a file"])

    url, upload = "", None
    if mode == "YouTube link":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    else:
        upload = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "mov"],
        )

    language = st.selectbox(
        "Spoken language",
        ["English", "Hinglish"],
        help="English uses local Whisper. Hinglish uses Sarvam to transcribe Hindi/Indian-language speech into English and requires SARVAM_API_KEY.",
    )

    run = st.button("Analyze recording", use_container_width=True, type="primary")

    if st.session_state.result:
        st.divider()
        if st.button("Start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------- run
if run:
    st.session_state.result = None
    st.session_state.messages = []
    source = None
    tmp_path = None
    if mode == "YouTube link":
        source = url.strip()
        if not source:
            st.sidebar.error("Paste a YouTube URL first.")
    else:
        if upload is None:
            st.sidebar.error("Choose a file to upload first.")
        else:
            suffix = os.path.splitext(upload.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = tmp.name
            source = tmp_path

    if source:
        try:
            st.session_state.result = analyze(source, language.lower())
            st.session_state.messages = []
        except Exception as exc:  # surface any pipeline error in the UI
            st.error(f"Couldn't process this recording: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# ---------------------------------------------------------------- main view
result = st.session_state.result

if not result:
    st.markdown(
        """
<div class="empty">
    <h2>Add a recording to get started</h2>
    <p>Paste a YouTube link or upload a file in the sidebar.<br>
    You'll get a summary, action items, decisions, open questions, and a chat that answers from the transcript.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

words = len(result["transcript"].split())
st.markdown(
    f"""
<div class="hero">
    <p class="hero-title">{html.escape(str(result['title']))}</p>
    <p class="hero-sub">{words:,} words transcribed</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Words", f"{words:,}")
c2.metric("Reading time", f"{max(1, round(words / 220))} min")
c3.metric("Action items", len([l for l in as_text(result["action_items"]).splitlines() if l.strip()]))

st.write("")

tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
    ["Summary", "Action items", "Decisions", "Open questions", "Transcript", "Chat"]
)

with tab_summary:
    panel(result["summary"], "No summary was generated.")
    st.download_button(
        "Download full report (.md)",
        data=build_report(result),
        file_name="meeting-report.md",
        mime="text/markdown",
    )

with tab_actions:
    panel(result["action_items"], "No action items were found in this recording.")

with tab_decisions:
    panel(result["key_decisions"], "No decisions were found in this recording.")

with tab_questions:
    panel(result["open_questions"], "No open questions were found in this recording.")

with tab_transcript:
    st.text_area("Transcript", result["transcript"], height=480, label_visibility="collapsed")
    st.download_button(
        "Download transcript (.txt)",
        data=result["transcript"],
        file_name="transcript.txt",
        mime="text/plain",
    )

with tab_chat:
    if not st.session_state.messages:
        st.caption("Ask anything about the recording, for example: “What did we decide about the deadline?”")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask about the recording"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Looking through the transcript…"):
                try:
                    answer = ask_question(result["rag_chain"], question)
                except Exception as exc:
                    answer = f"I couldn't answer that: {exc}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
