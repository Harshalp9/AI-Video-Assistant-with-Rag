import os

from core.summarizer import generate_title, summarize
from dotenv import load_dotenv

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.transcriber import transcribe_all
from utils.audio_processor import cleanup_audio, process_input

load_dotenv()

def run_pipeline(source :str, language :str = "english") -> dict:
    language = language.strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("language must be 'english' or 'hinglish'")
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    if language == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("SARVAM_API_KEY is required when language is 'hinglish'.")
    print("starting AI Video Assistant")

    chunks = process_input(source)

    try:
        transcript = transcribe_all(chunks, language)
    finally:
        cleanup_audio(chunks, source)
    if not transcript.strip():
        raise RuntimeError("Transcription returned no text; check the input audio and transcription setup.")
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Spoken language (english/hinglish; use hinglish for Hindi speech): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
