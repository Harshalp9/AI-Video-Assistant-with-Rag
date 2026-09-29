import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "downloades"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_youtube_audio(url :str) ->str:
    output_path = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # yt-dlp may select different source containers; the postprocessor
            # writes the converted file using the same stem and a .wav suffix.
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc)
        if "ffmpeg" in message.lower() or "ffprobe" in message.lower():
            raise RuntimeError(
                "FFmpeg and ffprobe are required to download/convert audio. "
                "Install FFmpeg and make sure both executables are on PATH."
            ) from exc
        raise
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"YouTube audio download did not produce a WAV file: {filename}")
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input media file not found: {input_path}")
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero")
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    source = source.strip()
    if not source:
        raise ValueError("Provide a YouTube URL or a local audio/video file path.")
    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


def cleanup_audio(chunks: list[str], source: str) -> None:
    """Remove generated chunks and intermediate audio after transcription."""
    for chunk_path in chunks:
        try:
            os.remove(chunk_path)
        except FileNotFoundError:
            pass

    if chunks:
        wav_path = chunks[0].rsplit("_chunk_", 1)[0]
        try:
            os.remove(wav_path)
        except FileNotFoundError:
            pass

    if not source.startswith(("http://", "https://")):
        converted_path = os.path.splitext(source)[0] + "_converted.wav"
        try:
            os.remove(converted_path)
        except FileNotFoundError:
            pass
