import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR=str(Path(__file__).resolve().parent.parent / "vector_db")
COLLECTION_NAME="meeting_transcript"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":'cpu'}
    )
    
def build_vector_store(transcript:str)->Chroma:
    print("Building vector Store")
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks=splitter.split_text(transcript)
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty transcript.")
    
    docs=[
        Document(page_content=chunk,metadata={'chunk_index':i})
        for i, chunk in enumerate(chunks)
    ]
    
    embeddings=get_embeddings()
    # Isolate each meeting so a later chat cannot retrieve passages from an
    # earlier recording stored in the persistent database.
    collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
    vector_store=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )
    
    return vector_store

def load_vector_store(collection_name: str)->Chroma:
    embeddings=get_embeddings()
    vector_store=Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    return vector_store

def get_retriever(vector_store:Chroma,k:int=4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k":k}
    )
