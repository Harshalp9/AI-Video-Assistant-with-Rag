# AI Video Meeting Assistant

Turn a meeting recording into a transcript, concise summary, action items, key decisions, and open questions. Then ask questions about the meeting using transcript-grounded chat.

The app accepts a YouTube URL or an audio/video file. English speech is transcribed locally with Whisper. Hinglish/Hindi speech can be transcribed and translated into English through Sarvam AI.

## Features

- Analyze YouTube recordings or uploaded audio/video files.
- Transcribe English locally with OpenAI Whisper.
- Transcribe and translate Hindi/Hinglish speech with Sarvam AI.
- Generate a meeting title and summary, and extract action items, decisions, and open questions with Gemini.
- Chat with the transcript using retrieval-augmented generation (RAG).
- Download the report as Markdown and the transcript as plain text.

## Requirements

- macOS, Linux, or Windows with Python 3.10 or newer.
- FFmpeg installed and available on `PATH`. On macOS with Homebrew:

  ```zsh
  brew install ffmpeg
  ```

- A Gemini API key for meeting analysis and chat.
- A Sarvam API key only if using Hinglish transcription.

The first run may download the Whisper model and the Hugging Face embedding model, so allow time and disk space for those downloads. Whisper inference and transcript embeddings run locally; Gemini and Sarvam features send content to their respective APIs.

## Setup

Run these commands from the project directory (the directory containing `requirements.txt`):

```zsh
cd "$HOME/Downloads/AI Video Assistant with Rag"
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternatively, use Python's built-in virtual environment and pip:

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## API keys

Create a `.env` file in the project root, alongside `app.py`, and add your keys:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_MAX_OUTPUT_TOKENS=1024
SARVAM_API_KEY=your_sarvam_api_key
```

`GEMINI_API_KEY` is required for all analyses. `GEMINI_MODEL` is optional; it defaults to `gemini-3.5-flash-lite`. Each model request is limited to 6,000 input characters, and generated output is capped at 1,024 tokens. `GEMINI_MAX_OUTPUT_TOKENS` can lower the output cap but cannot raise it. `SARVAM_API_KEY` is only required when selecting Hinglish. Keep `.env` private and do not commit API keys to source control.

## Run

Start the Streamlit interface from the project directory with the virtual environment activated:

```zsh
streamlit run app.py
```

Open the local URL printed by Streamlit. Select **YouTube link** or **Upload a file**, choose the spoken language, and select **Analyze recording**. Supported uploads are MP3, WAV, M4A, MP4, MKV, WEBM, and MOV.

The command-line interface is also available:

```zsh
python main.py
```

It prompts for a YouTube URL or local file path and a language (`english` or `hinglish`), prints the generated notes, and then provides an interactive transcript Q&A loop. Type `exit`, `quit`, or `q` to leave chat.

## Project layout

```text
app.py                  Streamlit user interface
main.py                 Command-line interface
core/
  extractor.py          Action items, decisions, and questions
  rag_engine.py         Transcript question-answering chain
  summarizer.py         Meeting title and summary generation
  transcriber.py        Whisper and Sarvam transcription
  vector_store.py       Local Chroma vector store and embeddings
utils/
  audio_processor.py    YouTube download, media conversion, and chunking
requirements.txt        Python dependencies
```

Generated Chroma data is stored in `vector_db/`. YouTube downloads and temporary audio are handled by the audio-processing pipeline.

## Troubleshooting

- **`requirements.txt` or `main.py` not found:** change to the project directory before running project commands. `uv` and Python resolve relative file paths from the current directory.
- **FFmpeg/ffprobe errors:** install FFmpeg and confirm `ffmpeg` and `ffprobe` are available on `PATH`.
- **Missing Gemini key:** add `GEMINI_API_KEY` to the project-root `.env` file.
- **Hinglish key error:** add `SARVAM_API_KEY` to `.env` or select English transcription.
- **Slow first run:** Whisper and sentence-transformer models may need to be downloaded before local transcription and indexing can begin.