from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import MAX_INPUT_CHARS, get_llm
from core.llm_limits import group_within_limit


def split_transcript(transcript: str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    
    return splitter.split_text(transcript)

def summarize(transcript: str, api_key: str | None = None) -> str:
    if not transcript.strip():
        raise ValueError("Cannot summarize an empty transcript.")
    llm = get_llm(api_key=api_key)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely."
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    while len("\n\n".join(chunk_summaries)) > MAX_INPUT_CHARS:
        groups = group_within_limit(chunk_summaries)
        if len(groups) >= len(chunk_summaries):
            return "\n\n".join(chunk_summaries)
        chunk_summaries = [
            map_chain.invoke({"text": group})
            for group in groups
        ]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert meeting summarizer. "
                    "Combine these partial summaries into one final "
                    "professional meeting summary in bullet points."
                )
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript: str, api_key: str | None = None) -> str:
    if not transcript.strip():
        raise ValueError("Cannot generate a title for an empty transcript.")
    llm = get_llm(api_key=api_key)

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Based on the meeting transcript, generate a short "
                        "professional meeting title (max 8 words). "
                        "Only return the title, nothing else."
                    )
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])
