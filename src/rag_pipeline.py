#!/usr/bin/env python3
"""
rag_pipeline.py

Usage:
    # Ingest a document into Pinecone
    python rag_pipeline.py ingest path/to/document.txt

    # Query the RAG system
    python rag_pipeline.py query "What is X about?"
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
load_dotenv()


# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "langchain-rag")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in environment")

# ---------------------------
# Initialize Pinecone client 
# ---------------------------
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# -----------------------
# Ensure index exists
# -----------------------
if 'youtuberag' not in pc.list_indexes().names():
    pc.create_index(
        name='youtuberag',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# -----------------------
# Helpers
# -----------------------
def init_pinecone(index_name: str, embeddings) -> None:
    """
    Ensure Pinecone index exists. If not, create it using the embedding dimension.
    """
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    if index_name in index_names:
        print(f"[pinecone] index '{index_name}' already exists.")
        return

    # get embedding dimension by calling embed on a small text
    vec = embeddings.embed_query("hello")
    dim = len(vec)
    print(f"[pinecone] creating index '{index_name}' with dimension={dim}")
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ------ load document from file ------
def load_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_text(encoding="utf-8")

# ------ chunk text into documents ------
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Use LangChain RecursiveCharacterTextSplitter to produce chunks.
    Returns list of langchain.schema.Document objects with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(text)
    docs = []
    for i, t in enumerate(texts):
        docs.append(Document(page_content=t, metadata={"chunk": i}))
    return docs

# ------ genrate embedding and upsert documents into pinecone ------
def upsert_documents(index_name: str, docs: List[Document], embeddings) -> None:
    """
    Upsert documents into Pinecone in batches.
    Each Document has .page_content and .metadata
    """
    BATCH = 32
    index = pc.Index(index_name)

    # build vectors in batches
    to_upsert = []
    for i, doc in enumerate(docs):
        vec = embeddings.embed_query(doc.page_content)
        # unique id per chunk
        uid = doc.metadata.get("id") or f"doc-{int(time.time())}-{i}"
        meta = doc.metadata or {}
        to_upsert.append((uid, vec, meta))

        if len(to_upsert) >= BATCH:
            index.upsert(vectors=to_upsert)
            to_upsert = []
    if to_upsert:
        index.upsert(vectors=to_upsert)

    print(f"[pinecone] upserted {len(docs)} chunks to index '{index_name}'.")

    
# -----------------------
# Query rewriting using LLM
# -----------------------
def rewrite_query(original_query: str, context_example: str = "") -> str:
    """
    Use the LLM to rewrite/expand the user's query into a better retrieval query.
    This is NOT fine-tuning the model; it's "query refinement".
    """
    # You can tune temperature/size as needed
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")  # or "gpt-4o-mini" / "gpt-4o" if available
    prompt = PromptTemplate(
        template=(
            "You are an assistant that rewrites user questions into an improved, "
            "concise retrieval query. Keep intent & important keywords. "
            "User question: {question}\n\nProduce a short improved query."
        ),
        input_variables=["question"],
    )
    query = prompt.format(question=original_query)
    resp = llm.invoke(query)
    return resp.content if hasattr(resp, "content") else str(resp)

# -----------------------
# Retrieval + Answering
# -----------------------
def query_rag(index_name: str, raw_query: str, top_k: int = 5) -> str:
    """
    Steps:
     1) rewrite query using LLM
     2) embed rewritten query and fetch top-k matches from pinecone
     3) pass retrieved context into LLM (RetrievalQA style) and return answer
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # initialize pinecone if needed
    init_pinecone(index_name, embeddings)

    # 1) rewrite / refine
    try:
        refined = rewrite_query(raw_query)
        print("[query] refined query:", refined)
    except Exception as e:
        print("[query] query rewrite failed, using original:", e)
        refined = raw_query

    # 2) fetch top_k from pinecone directly
    index = pc.Index(index_name)

    qvec = embeddings.embed_query(refined)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True, include_values=False)

    # build context from matches
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    context_pieces = []
    for m in matches:
        # Pinecone returns metadata from upsert; try to include 'text' in metadata if you saved it
        meta = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        text = meta.get("text") or meta.get("content") or meta.get("chunk_text") or ""
        # If you didn't store text in metadata, you should store chunk text or reference; here we assume metadata contains it.
        context_pieces.append(text or json.dumps(meta))

    # Fallback: if metadata did not hold text, we can fetch from your source store (not implemented here)

    # 3) Ask the LLM using the retrieved context
    chat = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    prompt_template = (
        "You are a helpful assistant. Use the following retrieved documents as context to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nProvide a concise, accurate answer and cite context chunks if applicable."
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    context_joined = "\n\n".join(context_pieces) if context_pieces else "No context found."
    input_text = prompt.format(context=context_joined, question=raw_query)
    resp = chat.invoke(input_text)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer

# -----------------------
# Ingest pipeline
# -----------------------
def ingest_document(path: str, index_name: str, chunk_size=1000, chunk_overlap=200):
    text = load_document(path)
    docs = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # attach content into metadata so we can retrieve it later
    for i, d in enumerate(docs):
        d.metadata["text"] = d.page_content
        d.metadata["source"] = str(path)
        d.metadata["id"] = f"{Path(path).stem}-chunk-{i}"

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    init_pinecone(index_name, embeddings)
    upsert_documents(index_name, docs, embeddings)
    print("[ingest] finished.")

# -----------------------
# CLI
# -----------------------
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
        ans = query_rag(PINECONE_INDEX, query, top_k=5)
        print("\n=== ANSWER ===\n")
        print(ans)
    else:
        print("Unknown command:", cmd)

if __name__ == "__main__":
    main()
