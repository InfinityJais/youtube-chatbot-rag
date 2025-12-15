from typing import List
from pathlib import Path
from config import pc, llm, embeddings, PINECONE_INDEX
from pinecone import ServerlessSpec
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def ensure_index_exists(index_name: str = PINECONE_INDEX, dimension: int = 1536) -> None:
    """Ensure Pinecone index exists or create it."""
    index_names = pc.list_indexes().names() 
    
    if index_name in index_names:
        return

    print(f"[pinecone] Creating index '{index_name}' with dimension={dimension}...")
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)
    print(f"[pinecone] Index '{index_name}' created.")


def format_docs(docs: List[Document]) -> str:
    """Formats the list of retrieved Documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(index_name: str = PINECONE_INDEX):
    """Constructs the full RAG chain using LCEL."""
    
    ensure_index_exists(index_name)

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    RAG_PROMPT_TEMPLATE = """
    You are a helpful assistant. Use the following retrieved documents as context to answer the question. 
    If the question cannot be answered from the provided context, state that clearly and concisely.

    Context:{context}
    Question: {question}

    Provide a concise, accurate answer and cite the source or chunk number from the context if possible.
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Ingestion Helper ---
def ingest_chunks_to_pinecone(transcript_path: Path, index_name: str = PINECONE_INDEX):
    """Alternative: Load existing transcript file and index it."""
    from ingestion_service import _chunk_and_enrich
    
    ensure_index_exists(index_name)
    
    text = transcript_path.read_text(encoding="utf-8")
    metadata = {
        "source": str(transcript_path),
        "file_type": "transcript"
    }
    
    docs = _chunk_and_enrich(text, transcript_path, metadata)
    
    PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=index_name
    )
    
    print(f"[RAG] Successfully ingested {len(docs)} chunks from file.")