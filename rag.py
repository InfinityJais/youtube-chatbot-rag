import os
import logging
from typing import List, Dict
from pathlib import Path

from pinecone import ServerlessSpec
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    from config import pc, llm, embeddings, PINECONE_INDEX
except ImportError:
    # Fallback/Error if config is missing
    raise ImportError("Could not import config.py. Make sure pc, llm, embeddings, and PINECONE_INDEX are defined.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Index Management ---

def ensure_index_exists(index_name: str = PINECONE_INDEX, dimension: int = 1536):
    """Checks if index exists, creates it if not."""
    try:
        # pc.list_indexes() returns an object, .names() gives the list
        existing_indexes = [i.name for i in pc.list_indexes()]

        if index_name in existing_indexes:
            logger.info(f"Index '{index_name}' already exists")
            return

        logger.info(f"Creating index '{index_name}'...")
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=spec
        )
        logger.info(f"Index '{index_name}' created successfully")
        
    except Exception as e:
        logger.error(f"Failed to ensure index exists: {e}")
        raise

def clear_index_data(index_name: str = PINECONE_INDEX, force: bool = False):
    """Deletes ALL vectors from the index but keeps the index intact."""
    try:
        existing_indexes = [i.name for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.warning(f"Index '{index_name}' does not exist, nothing to clear.")
            return

        if not force:
            logger.warning(f"Index '{index_name}' exists. Data deletion skipped. Set force=True to clear.")
            return

        index = pc.Index(index_name)
        index.delete(delete_all=True)
        logger.info(f"All data cleared from index '{index_name}'")
        
    except Exception as e:
        logger.error(f"Error clearing index: {e}")

def index_stats(index_name: str = PINECONE_INDEX):
    """Fetch index statistics and basic health info."""
    try:
        existing_indexes = [i.name for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.error(f"Index '{index_name}' does not exist")
            return

        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        vector_count = stats.get("total_vector_count", 0)
        namespaces = list(stats.get('namespaces', {}).keys())

        print("\n📊 Index Stats")
        print("-------------------")
        print(f"Index name        : {index_name}")
        print(f"Total vectors     : {vector_count}")
        print(f"Namespaces        : {namespaces or 'default'}")
        
        if vector_count == 0:
            print("Health status     : EMPTY ")
        else:
            print("Health status     : HEALTHY ")
        print("-------------------\n")

        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")

# --- Processing & Ingestion ---

def _chunk_and_enrich(text: str, source_path: Path, metadata: dict = None) -> List[Document]:
    """Split text into chunks and create Document objects with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc_metadata = {
            "source": str(source_path.name), # Just filename is usually cleaner
            "chunk_index": i,
            "total_chunks": len(chunks),
            **(metadata or {})
        }
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents

def ingest_chunks_to_pinecone(transcript_path: Path, index_name: str = PINECONE_INDEX):
    """Reads a transcript file, chunks it, and uploads to Pinecone."""
    if not transcript_path.exists():
        logger.error(f"File not found: {transcript_path}")
        return

    try:
        text = transcript_path.read_text(encoding="utf-8")
        metadata = {
            "source_path": str(transcript_path),
            "file_type": "transcript"
        }
        
        # Use the local function defined above
        docs = _chunk_and_enrich(text, transcript_path, metadata)
        
        if not docs:
            logger.warning(f"No text found in {transcript_path.name}")
            return

        logger.info(f"Ingesting {len(docs)} chunks from {transcript_path.name}...")
        
        PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=index_name
        )
        logger.info(f"Successfully ingested {transcript_path.name} ")
        
    except Exception as e:
        logger.error(f"Failed to ingest {transcript_path.name}: {e}")

# --- RAG Chain Construction ---

def format_docs(docs: List[Document]) -> str:
    """Formats the list of retrieved Documents into a single string."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" 
        for doc in docs
    )

def get_rag_chain(index_name: str = PINECONE_INDEX):
    """Constructs the full RAG chain using LCEL."""
    
    # Ensure index exists before building chain to avoid runtime errors
    ensure_index_exists(index_name)

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # k=5 means retrieve top 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    RAG_PROMPT_TEMPLATE = """
    You are a helpful assistant. Use the following retrieved documents as context to answer the question. 
    If the question cannot be answered from the provided context, state that clearly and concisely.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup Index
    ensure_index_exists()
    
    # Optional: Clear data if you want a fresh start
    # clear_index_data(force=True)

    # 2. Ingest Data
    transcripts_dir = Path("./transcripts")
    
    if transcripts_dir.exists():
        txt_files = list(transcripts_dir.glob("*.txt"))
        if txt_files:
            logger.info(f"Found {len(txt_files)} transcripts. Starting ingestion...")
            for path in txt_files:
                ingest_chunks_to_pinecone(path)
        else:
            logger.warning("No .txt files found in ./transcripts directory.")
    else:
        logger.warning("Directory './transcripts' does not exist. Skipping ingestion.")

    # 3. Check Stats
    index_stats() 
    
    # 4. Test RAG
    logger.info("Initializing RAG Chain...")
    rag_chain = get_rag_chain()
    
    test_question = "What is this video about explain me in 1 line?"
    print(f"\n❓ Question: {test_question}")
    
    try:
        answer = rag_chain.invoke(test_question)
        print(f"💡 Answer:\n{answer}\n")
    except Exception as e:
        logger.error(f"RAG Chain failed: {e}")