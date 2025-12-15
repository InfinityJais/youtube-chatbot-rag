import os
import sys
import json
import time
import shutil
import subprocess
import random
from pathlib import Path
from typing import List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from operator import itemgetter 

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl

from dotenv import load_dotenv
load_dotenv()

# --- Third-Party Imports ---
import yt_dlp
from langsmith import Client
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI # Used for direct Whisper call

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION & CLIENTS
# ----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY in environment")

# API Service Config
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "youtube-rag-index")
TEMP_DIR = Path("tmp_video_files")

# Initialize Global Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
ls_client = Client()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# LLM and Embeddings
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# FastAPI App
app = FastAPI(title="YouTube RAG Pipeline", version="1.0")


# --- Pydantic Models for API ---
class IngestRequest(BaseModel):
    url: str
    
class QueryRequest(BaseModel):
    query: str
    

# ----------------------------------------------------------------------
# YOUTUBE DOWNLOAD & TRANSCRIPTION LOGIC (from script 1)
# ----------------------------------------------------------------------

def _get_video_info(youtube_url: str, out_dir: Path) -> Tuple[Path, dict]:
    """Downloads audio and returns path to WAV and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id") or info.get("url")
        # Find the path of the downloaded file (assuming yt-dlp's default naming)
        downloaded = next(out_dir.glob(f"{video_id}.*"), None)
        if downloaded is None:
             raise RuntimeError(f"Could not locate downloaded file for {video_id}")
    
    # 1. Convert to 16k mono WAV using ffmpeg
    wav_path = out_dir / f"{video_id}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(downloaded),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(downloaded) # Clean up original downloaded file
    return wav_path, info

def _get_duration_seconds(file_path: Path) -> float:
    """Uses ffprobe to get audio duration."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", 
           "-of", "json", str(file_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    duration = float(info['streams'][0]['duration'])
    return duration

def _split_and_transcribe_wav(wav_path: Path, title: str) -> Path:
    """Splits WAV and transcribes chunks in parallel."""
    CHUNKS_DIR = TEMP_DIR / "chunks"
    if CHUNKS_DIR.exists(): shutil.rmtree(CHUNKS_DIR)
    CHUNKS_DIR.mkdir()
    
    CHUNK_LENGTH_S = 180
    OVERLAP_S = 2
    MAX_WORKERS = 3
    MAX_RETRIES = 3
    
    total = _get_duration_seconds(wav_path)
    step = CHUNK_LENGTH_S - OVERLAP_S
    chunks_to_process = []
    start = 0.0
    idx = 0
    
    # Split audio file into small chunks (no memory load)
    while start < total:
        end = min(start + CHUNK_LENGTH_S, total)
        out_file = CHUNKS_DIR / f"chunk_{idx:03d}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start}", "-i", str(wav_path), "-t", f"{end - start}",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks_to_process.append((out_file, start, end))
        idx += 1
        start += step

    # Parallel Transcription
    results = []
    def transcribe_worker(chunk_info):
        chunk_path, start_s, end_s = chunk_info
        # Use the global openai_client directly
        for attempt in range(MAX_RETRIES):
            try:
                with open(chunk_path, "rb") as f:
                    resp = openai_client.audio.translations.create(model="whisper-1", file=f)
                return {"start": start_s, "end": end_s, "text": resp.text.strip()}
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"FAILED CHUNK {chunk_path.name} after {MAX_RETRIES} retries.")
                    return {"start": start_s, "end": end_s, "text": f"**TRANSCRIPTION ERROR**", "file": str(chunk_path), "error": str(e)}
                time.sleep((2 ** attempt) + random.random())
        return {"start": start_s, "end": end_s, "text": f"**TRANSCRIPTION ERROR**"}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(transcribe_worker, c) for c in chunks_to_process]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Stitch and Save Transcript
    results.sort(key=lambda r: r["start"])
    stitched_text = "\n\n".join(r["text"] for r in results if not r["text"].startswith('**TRANSCRIPTION ERROR**'))
    
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_")).rstrip()[:100]
    output_path = Path("transcripts") / f"{safe_title}_translated.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(stitched_text, encoding="utf-8")

    shutil.rmtree(CHUNKS_DIR) # Clean up chunks
    return output_path

# ----------------------------------------------------------------------
# RAG PIPELINE CORE LOGIC (from script 2)
# ----------------------------------------------------------------------

def _ensure_index_exists(index_name: str, dimension: int = 1536) -> None:
    """Ensure Pinecone index exists or create it."""
    # FIX: Correctly call names() method
    index_names = pc.list_indexes().names() 
    
    if index_name in index_names:
        return

    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)


def _chunk_and_enrich(text: str, file_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Uses LangChain splitter and enriches Document metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    
    texts = splitter.split_text(text)
    for i, t in enumerate(texts):
        meta = {
            "chunk": i, 
            "text": t,
            "source": str(file_path.name), 
            "id": f"{file_path.stem}-chunk-{i}"
        }
        docs.append(Document(page_content=t, metadata=meta))
    return docs


def _ingest_document_to_pinecone(transcript_path: Path, index_name: str):
    """Loads text from file, chunks, embeds, and upserts to Pinecone."""
    
    _ensure_index_exists(index_name)
    
    text = transcript_path.read_text(encoding="utf-8")
    docs = _chunk_and_enrich(text, transcript_path)
    
    # Upsert using LangChain's PineconeVectorStore utility
    PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=index_name
    )
    
    os.remove(transcript_path) # Clean up the final transcript file
    print(f"Successfully ingested {len(docs)} chunks into '{index_name}'.")


def get_rag_chain(index_name: str):
    """Constructs the full RAG chain using LCEL."""
    
    # Ensure index exists before trying to access it
    _ensure_index_exists(index_name)

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

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

# ----------------------------------------------------------------------
# BACKGROUND TASK CHAIN FUNCTION
# This is the function that will be called by FastAPI's BackgroundTasks
# ----------------------------------------------------------------------

def download_transcribe_and_ingest(url: str, index_name: str):
    """
    Main synchronous function to execute the full pipeline.
    This runs in a background thread provided by FastAPI.
    """
    try:
        if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir()

        print(f"[INGEST] Starting pipeline for URL: {url}")
        
        # 1. Download, Convert, and Get Info
        wav_path, info = _get_video_info(url, TEMP_DIR)
        print(f"[INGEST] Downloaded and converted WAV: {wav_path.name}")

        # 2. Split and Transcribe (Whisper)
        transcript_path = _split_and_transcribe_wav(wav_path, info.get("title", "unknown_video"))
        print(f"[INGEST] Transcription saved to: {transcript_path.name}")
        
        # 3. Clean up the WAV file
        os.remove(wav_path)

        # 4. Ingest into Pinecone Vector Store
        _ingest_document_to_pinecone(transcript_path, index_name)
        
        print(f"[INGEST] Pipeline SUCCESS for {url}")

    except Exception as e:
        print(f"[INGEST] Pipeline FAILED for {url}. Error: {e}")
        traceback.print_exc(file=sys.stdout)
    finally:
        # Final cleanup of temp directory
        if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)


# ----------------------------------------------------------------------
# FASTAPI ENDPOINTS
# ----------------------------------------------------------------------

@app.post("/ingest-video", status_code=202)
async def start_ingestion_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the long-running video ingestion process.
    Returns 202 Accepted immediately.
    """
    url = request.url
    
    # Offload the synchronous, long-running job to a background thread
    background_tasks.add_task(
        download_transcribe_and_ingest, 
        url, 
        PINECONE_INDEX
    )
    
    return {
        "status": "Processing started",
        "message": f"Ingestion for {url} has begun in the background. Check server logs for completion status."
    }


@app.post("/query")
async def ask_query_endpoint(request: QueryRequest):
    """
    Endpoint to ask a question against the RAG system.
    """
    try:
        # 1. Check if the index exists before starting the RAG chain
        if PINECONE_INDEX not in pc.list_indexes().names():
             raise HTTPException(
                status_code=404, 
                detail=f"Vector Index '{PINECONE_INDEX}' not found. Please ingest a video first."
            )

        # 2. Initialize and invoke the LCEL RAG chain
        rag_chain = get_rag_chain(PINECONE_INDEX)
        answer = rag_chain.invoke(request.query)
        
        return {
            "query": request.query,
            "answer": answer
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        # Catch errors from Pinecone connection, LLM call, etc.
        raise HTTPException(
            status_code=503, 
            detail=f"RAG query failed. Error: {e}"
        )