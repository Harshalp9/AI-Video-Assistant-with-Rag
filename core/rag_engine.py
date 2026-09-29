import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from core.vector_store import build_vector_store, get_retriever


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )
    
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):
    if not transcript.strip():
        raise ValueError("Cannot build a chat index from an empty transcript.")
    
    vector_store=build_vector_store(transcript)
    
    retriever=get_retriever(vector_store, k=4)
    
    llm=get_llm()
    
    prompt=ChatPromptTemplate.from_messages(
        
        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say exactly:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human","{question}"),
        ]
    )
    
    #full LCEL Rag pipeline
    
    rag_chain=(
        
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
        |prompt|llm|StrOutputParser()
    )
    
    return rag_chain

def ask_question(rag_chain,question:str)->str:
    if not question.strip():
        raise ValueError("Enter a question about the recording.")
    print(f"Question:{question}")
    answer=rag_chain.invoke(question)
    print(f"answer:{answer}")
    return answer
