from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import MAX_INPUT_CHARS


def split_for_model(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_INPUT_CHARS,
        chunk_overlap=200,
    )
    return splitter.split_text(text)


def group_within_limit(texts: list[str]) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_size = 0

    bounded_texts = [
        text[start : start + MAX_INPUT_CHARS]
        for text in texts
        for start in range(0, len(text), MAX_INPUT_CHARS)
    ]
    for text in bounded_texts:
        added_size = len(text) + (2 if current else 0)
        if current and current_size + added_size > MAX_INPUT_CHARS:
            groups.append("\n\n".join(current))
            current = []
            current_size = 0
            added_size = len(text)
        current.append(text)
        current_size += added_size

    if current:
        groups.append("\n\n".join(current))
    return groups