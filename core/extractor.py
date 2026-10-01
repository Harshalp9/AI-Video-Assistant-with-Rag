#Actionableitems , decision , questions 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm
from core.llm_limits import group_within_limit, split_for_model


def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_in_chunks(transcript: str, instruction: str, empty_result: str) -> str:
    extract_chain = build_chain(instruction)
    results = [extract_chain.invoke(chunk) for chunk in split_for_model(transcript)]
    if not results:
        return empty_result

    merge_chain = build_chain(
        "Combine and deduplicate these partial results. Preserve every distinct "
        f"item and use the requested format. {instruction}"
    )
    while len(results) > 1:
        groups = group_within_limit(results)
        if len(groups) >= len(results):
            return "\n\n".join(results).strip()
        results = [merge_chain.invoke(group) for group in groups]
    return results[0].strip() or empty_result


def extract_action_items(transcript:str)->str:
    if not transcript.strip():
        raise ValueError("Cannot extract action items from an empty transcript.")
    return extract_in_chunks(
        transcript,
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        "No action items found.",
    )


def extract_key_decisions(transcript: str) -> str:
    if not transcript.strip():
        raise ValueError("Cannot extract decisions from an empty transcript.")
    return extract_in_chunks(
        transcript,
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        "No key decisions found.",
    )


def extract_questions(transcript: str) -> str:
    if not transcript.strip():
        raise ValueError("Cannot extract questions from an empty transcript.")
    return extract_in_chunks(
        transcript,
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        "No open questions found.",
    )
