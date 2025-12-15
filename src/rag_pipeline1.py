import os
import sys
import json
import time
from pathlib import Path
from typing import List, Union
from operator import itemgetter # Used for RunnablePassthrough.assign

from dotenv import load_dotenv
load_dotenv()

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith Imports ---
from langsmith import Client
# Initialize LangSmith Client (requires LANGCHAIN_API_KEY to be set)
ls_client = Client()


# --- Pinecone Imports ---
from pinecone import Pinecone, ServerlessSpec


# -----------------------
# CONFIG & INITIALIZATION
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "langchain-rag")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY in environment")

# Initialize Pinecone client (Global)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define embedding model (Used in both ingest and query)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# -----------------------
# HELPER FUNCTIONS (Ingestion/Pre-processing)
# -----------------------

def ensure_index_exists(index_name: str, dimension: int = 1536) -> None:
    """Ensure Pinecone index exists or create it."""
    index_names = pc.list_indexes().names
    
    if index_name in index_names:
        print(f"[pinecone] Index '{index_name}' already exists.")
        return

    print(f"[pinecone] Creating index '{index_name}' with dimension={dimension}...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"[pinecone] Index '{index_name}' created.")

# Define directory for transcripts

"""
def load_document(path: str) -> str:
    #Loads raw text content from a file path.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_text(encoding="utf-8")
"""
def load_document() -> str:
    """Loads the single txt file from the transcripts folder."""
    transcripts_dir = Path("transcripts")
    
    txt_files = list(transcripts_dir.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError("No .txt file found in transcripts folder")
    
    txt_file = txt_files[0]
    return txt_file.read_text(encoding="utf-8")


def chunk_text(text: str, path: Union[str, Path], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Uses LangChain splitter and enriches Document metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    
    # We split the text, then manually create Documents with rich metadata
    texts = splitter.split_text(text)
    for i, t in enumerate(texts):
        # Attach content into metadata for easy retrieval later (Crucial for this pipeline)
        meta = {
            "chunk": i, 
            "text": t,
            "source": str(Path(path)),
            "id": f"{Path(path).stem}-chunk-{i}"
        }
        docs.append(Document(page_content=t, metadata=meta))
    
    return docs


def ingest_document(path: str, index_name: str):
    """Orchestrates document loading, chunking, and upserting."""
    
    # 1. Ensure index is ready
    ensure_index_exists(index_name)
    
    # 2. Load and Chunk
    text = load_document(path)
    docs = chunk_text(text)
    
    # 3. Initialize VectorStore and Upsert Documents
    # LangChain's PineconeVectorStore.from_documents handles the embedding and upsert
    print(f"[ingest] Upserting {len(docs)} chunks into '{index_name}'...")
    PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=index_name
    )
    
    print("[ingest] Finished upserting documents.")


# -----------------------
# RAG CHAIN DEFINITION (LCEL)
# -----------------------

# 1. Define LLM and Prompt Template
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

def format_docs(docs: List[Document]) -> str:
    """Formats the list of retrieved Documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following retrieved documents as context to answer the question. 
If the question cannot be answered from the provided context, state that clearly and concisely.

Context:{context}

Question: {question}

Provide a concise, accurate answer and cite the source or chunk number from the context if possible.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def get_rag_chain(index_name: str):
    """Constructs the full RAG chain using LCEL."""

    # Ensure index exists before trying to access it
    ensure_index_exists(index_name)

    # 1. Create Retriever from the existing Pinecone index
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# -----------------------
# CLI
# -----------------------

# --- RAG CHAIN EXECUTION ---
def query_rag(index_name: str, raw_query: str):
    """Invokes the RAG chain and prints the answer."""
    rag_chain = get_rag_chain(index_name)
    
    print("[query] Invoking RAG chain...")

    # Define a config object to store the Run ID
    config = {}

    # 1. Invoke the chain, passing the config object
    ans = rag_chain.invoke(raw_query, config)
    
    # 2. Extract the Run ID from the configuration after invocation
    run_id = config.get('callbacks')[0].latest_run.id
    
    print("\n=== ANSWER ===\n")
    print(ans)
    
    # 3. Use the Run ID to record feedback (Example)
    record_user_feedback(run_id)


def record_user_feedback(run_id):
    """Example function to record custom feedback."""
    global ls_client
    
    # --- Example Feedback Recording ---
    print("\n--- Recording Feedback ---")
    
    # Simulate user input/satisfaction (e.g., 1 for Good, 0 for Bad)
    user_score = input("Was the answer helpful? (y/n): ").lower()
    
    if user_score == 'y':
        # Record a passing score for evaluation metric 'helpful'
        ls_client.record_feedback(
            run_id=run_id,
            key="helpful",
            score=1,
            comment="User found the RAG answer useful."
        )
        print("Feedback recorded: Helpful (Score: 1)")
    else:
        ls_client.record_feedback(
            run_id=run_id,
            key="helpful",
            score=0,
            comment="User found the RAG answer unhelpful."
        )
        print("Feedback recorded: Unhelpful (Score: 0)")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python rag_pipeline.py ingest <path-to-doc>")
        print("  python rag_pipeline.py query \"your question\"")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "ingest":
        path = sys.argv[2]
        ingest_document(path, PINECONE_INDEX)
    elif cmd == "query":
        query = sys.argv[2]
        query_rag(PINECONE_INDEX, query)
    else:
        print("Unknown command:", cmd)

if __name__ == "__main__":
    main()