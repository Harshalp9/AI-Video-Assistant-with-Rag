import os
from pathlib import Path

import requests
import whisper
from pydub import AudioSegment
from dotenv import load_dotenv

# Resolve .env relative to the project rather than the process working directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    model = load_model()

    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def _send_to_sarvam(piece_path: str, api_key: str | None = None) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    api_key = api_key or os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, api_key: str | None = None) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not (api_key or os.getenv("SARVAM_API_KEY")):
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path, api_key=api_key) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()





def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
    sarvam_api_key: str | None = None,
) -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    language = language.strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("language must be 'english' or 'hinglish'")
    if language == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, api_key=sarvam_api_key)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english",
    sarvam_api_key: str | None = None,
) -> str:
    if not chunks:
        raise ValueError("No audio chunks were provided for transcription.")
    if language.strip().lower() not in {"english", "hinglish"}:
        raise ValueError("language must be 'english' or 'hinglish'")

    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(
            chunk,
            language=language,
            sarvam_api_key=sarvam_api_key,
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
