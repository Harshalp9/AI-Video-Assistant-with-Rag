import os

from langchain_google_genai import ChatGoogleGenerativeAI

MAX_INPUT_CHARS = 6000
MAX_OUTPUT_TOKENS = 1024


def get_llm(temperature: float = 1.0) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment / .env")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=min(
            MAX_OUTPUT_TOKENS,
            max(1, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))),
        ),
    )