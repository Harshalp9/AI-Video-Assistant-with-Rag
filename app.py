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
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
    --ink: #173b35;
    --muted: #66766f;
    --paper: #f2f3eb;
    --card: #fffefa;
    --line: #d8ded2;
    --accent: #d65e45;
    --accent-soft: #f8e7df;
    --lime: #d1e76a;
    --forest: #173b35;
}

html, body, [class*="css"], .stMarkdown, .stTextInput input, .stChatInput textarea {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}
.stApp {
    background-color: var(--paper);
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 39px, rgba(23, 59, 53, .025) 40px
    );
}
.block-container { padding-top: 1.5rem; max-width: 1180px; }

h1, h2, h3, .hero-title {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    letter-spacing: 0;
    color: var(--ink);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--forest);
    border-right: 1px solid rgba(255, 255, 255, .12);
}
section[data-testid="stSidebar"] * { color: #e8eee4 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] h2 {
    font-size: 1.15rem;
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #b6c8b8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, .18);
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #224a41 !important;
    border-color: #43675a !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #224a41;
    border: 1px dashed #72917b;
    border-radius: 6px;
}
section[data-testid="stSidebar"] button {
    background: var(--lime) !important;
    border: none !important;
    border-radius: 6px !important;
    color: var(--forest) !important;
    font-weight: 700;
    transition: transform .18s ease, background-color .18s ease;
}
section[data-testid="stSidebar"] button:hover {
    background: #e0f18b !important;
    transform: translateY(-1px);
}
section[data-testid="stSidebar"] label p {
    font-size: .88rem;
    font-weight: 500;
}

/* workspace header */
.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: .25rem 0 1.1rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--line);
}
.brand-lockup {
    display: flex;
    align-items: center;
    gap: .7rem;
    color: var(--ink);
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.12rem;
    font-weight: 700;
}
.brand-mark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 6px;
    background: var(--forest);
    color: var(--lime);
    font-size: 1.05rem;
}
.workspace-tag, .eyebrow {
    color: var(--muted);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}

/* empty state and completed meeting */
.empty, .hero {
    position: relative;
    overflow: hidden;
    background: var(--forest);
    border-radius: 8px;
    color: #fffefa;
}
.empty {
    min-height: 390px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 3.2rem clamp(1.5rem, 6vw, 5rem);
    text-align: left;
}
.empty::after {
    content: "";
    position: absolute;
    right: -3rem;
    bottom: -8rem;
    width: 25rem;
    height: 25rem;
    border: 1px solid rgba(209, 231, 106, .22);
    border-radius: 50%;
    box-shadow: 0 0 0 30px rgba(209, 231, 106, .035),
                0 0 0 70px rgba(209, 231, 106, .025);
    pointer-events: none;
}
.empty .eyebrow, .hero .eyebrow { color: var(--lime); }
.empty h1 {
    max-width: 650px;
    margin: .8rem 0 .75rem;
    color: #fffefa;
    font-size: 4rem;
    line-height: 1;
}
.empty p {
    max-width: 470px;
    color: #c5d3c4;
    font-size: 1.05rem;
    line-height: 1.65;
}
.waveform {
    position: absolute;
    right: 7%;
    top: 50%;
    display: flex;
    align-items: center;
    gap: 5px;
    height: 88px;
    transform: translateY(-50%);
    opacity: .82;
}
.waveform span {
    width: 4px;
    height: var(--bar-height);
    border-radius: 2px;
    background: var(--lime);
    animation: breathe 2.8s ease-in-out infinite alternate;
    animation-delay: var(--delay);
}
@keyframes breathe {
    from { transform: scaleY(.72); opacity: .55; }
    to { transform: scaleY(1); opacity: 1; }
}
.hero {
    padding: 1.65rem 1.8rem;
    margin-bottom: 1rem;
    background-image: linear-gradient(110deg, #173b35 0%, #245247 100%);
}
.hero-title {
    max-width: 850px;
    margin: .45rem 0 .4rem;
    color: #fffefa;
    font-size: 2.15rem;
    font-weight: 700;
    line-height: 1.13;
    overflow-wrap: anywhere;
}
.hero-sub { color: #c5d3c4; margin: 0; }

/* stats */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: .85rem 1rem;
}
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.6rem;
}

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; font-size: .95rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* content panels */
.panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    line-height: 1.65;
}
.transcript {
    max-height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-size: .95rem;
}
div[data-testid="stDownloadButton"] button {
    border-color: var(--ink);
    border-radius: 6px;
    color: var(--ink);
    font-weight: 600;
}
div[data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

@media (max-width: 700px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .empty { min-height: 340px; padding: 2.2rem 1.5rem; }
    .empty h1 { max-width: 430px; font-size: 2.65rem; }
    .empty p { max-width: 390px; font-size: .98rem; }
    .waveform { right: 1.4rem; top: auto; bottom: 1.1rem; transform: scale(.72); transform-origin: bottom right; }
    .hero { padding: 1.3rem; }
    .hero-title { font-size: 1.8rem; }
    .workspace-tag { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
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


def analyze(
    source: str,
    language: str,
    gemini_api_key: str | None = None,
    sarvam_api_key: str | None = None,
) -> dict:
    language = language.strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("language must be 'english' or 'hinglish'")
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    sarvam_api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("Enter a Gemini API key in the sidebar to analyze a recording.")
    if language == "hinglish" and not sarvam_api_key:
        raise RuntimeError("Enter a Sarvam API key to transcribe Hinglish recordings.")
    with st.status("Working on your recording…", expanded=True) as status:
        st.write("Preparing audio")
        chunks = process_input(source)

        st.write("Transcribing")
        try:
            transcript = transcribe_all(
                chunks,
                language,
                sarvam_api_key=sarvam_api_key,
            )
        finally:
            cleanup_audio(chunks, source)
        if not transcript.strip():
            status.update(label="No speech found", state="error")
            raise RuntimeError(
                "Transcription returned no text. Check that the recording has "
                "audible speech and that your transcription setup is working."
            )

        st.write("Writing title and summary")
        title = generate_title(transcript, api_key=gemini_api_key)
        summary = summarize(transcript, api_key=gemini_api_key)

        st.write("Finding action items, decisions and questions")
        action_items = extract_action_items(transcript, api_key=gemini_api_key)
        decisions = extract_key_decisions(transcript, api_key=gemini_api_key)
        questions = extract_questions(transcript, api_key=gemini_api_key)

        st.write("Indexing the transcript for chat")
        rag_chain = build_rag_chain(transcript, api_key=gemini_api_key)

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

    gemini_api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Used for summaries, extracted notes, and transcript chat.",
    )
    sarvam_api_key = ""
    if language == "Hinglish":
        sarvam_api_key = st.text_input(
            "Sarvam API key",
            type="password",
            help="Used to transcribe and translate Hinglish audio.",
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
            st.session_state.result = analyze(
                source,
                language.lower(),
                gemini_api_key=gemini_api_key,
                sarvam_api_key=sarvam_api_key,
            )
            st.session_state.messages = []
        except Exception as exc:  # surface any pipeline error in the UI
            st.error(f"Couldn't process this recording: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# ---------------------------------------------------------------- main view
result = st.session_state.result
st.markdown(
    """
<div class="masthead">
    <div class="brand-lockup"><span class="brand-mark">M</span> Meeting Assistant</div>
    <div class="workspace-tag">Meeting studio &nbsp; / &nbsp; Private workspace</div>
</div>
""",
    unsafe_allow_html=True,
)

if not result:
    st.markdown(
        """
<div class="empty">
    <div class="eyebrow">A clearer view of the conversation</div>
    <h1>Make room for the ideas that matter.</h1>
    <p>Your meeting, brought into focus.</p>
    <div class="waveform" aria-hidden="true">
        <span style="--bar-height: 18px; --delay: 0ms"></span>
        <span style="--bar-height: 34px; --delay: 90ms"></span>
        <span style="--bar-height: 54px; --delay: 180ms"></span>
        <span style="--bar-height: 28px; --delay: 270ms"></span>
        <span style="--bar-height: 70px; --delay: 360ms"></span>
        <span style="--bar-height: 42px; --delay: 450ms"></span>
        <span style="--bar-height: 24px; --delay: 540ms"></span>
        <span style="--bar-height: 62px; --delay: 630ms"></span>
        <span style="--bar-height: 34px; --delay: 720ms"></span>
        <span style="--bar-height: 76px; --delay: 810ms"></span>
        <span style="--bar-height: 46px; --delay: 900ms"></span>
        <span style="--bar-height: 22px; --delay: 990ms"></span>
        <span style="--bar-height: 58px; --delay: 1080ms"></span>
        <span style="--bar-height: 32px; --delay: 1170ms"></span>
        <span style="--bar-height: 68px; --delay: 1260ms"></span>
        <span style="--bar-height: 40px; --delay: 1350ms"></span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

words = len(result["transcript"].split())
st.markdown(
    f"""
<div class="hero">
    <div class="eyebrow">Meeting brief</div>
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
