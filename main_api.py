import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
import re

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LOCAL MODULE IMPORTS ---
# Ensure these filenames match exactly what you saved earlier!
from config import PINECONE_INDEX, pc
from ingestion_service import transcribe_video_pipeline
from rag import get_rag_chain, ingest_chunks_to_pinecone, ensure_index_exists

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PYDANTIC MODELS ---
class IngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

class DeleteRequest(BaseModel):
    delete_index: bool = True
    delete_transcripts: bool = True
    delete_temp_workspace: bool = True
    index_name: Optional[str] = None  # Added this field so request.index_name works

# --- APP SETUP ---
app = FastAPI(
    title="YouTube Transcript RAG API",
    description="RAG system for YouTube video Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track status: { "video_id": "processing" | "ready" | "failed" }
ingestion_status = {}

def extract_video_id_server(url: str):
    """Simple helper to extract ID for status tracking key."""
    match = re.search(r"(?:v=|\/)([\w-]{11})(?:\?|&|\/|$)", url)
    return match.group(1) if match else "unknown"

# --- BACKGROUND TASK WRAPPER (The Glue) ---
def background_ingestion_task(youtube_url: str, index_name: str):
    """
    Orchestrates the full pipeline with status updates.
    """
    video_id = extract_video_id_server(youtube_url)
    ingestion_status[video_id] = "processing" # <--- SET STATUS TO PROCESSING

    logger.info(f"🚀 Starting background pipeline for: {youtube_url}")

    try:
        # Step 1: Transcribe Video
        transcript_path = transcribe_video_pipeline(youtube_url, working_dir="./temp_workspace")

        if not transcript_path:
            logger.error("❌ Transcription failed. Stopping pipeline.")
            ingestion_status[video_id] = "failed"
            return

        # Step 2: Ingest into Pinecone
        ensure_index_exists(index_name)
        ingest_chunks_to_pinecone(transcript_path, index_name)

        ingestion_status[video_id] = "ready" # <--- SET STATUS TO READY
        logger.info(f"✅ Pipeline complete for: {youtube_url}")

    except Exception as e:
        ingestion_status[video_id] = "failed"
        logger.error(f"💥 Critical error in background task: {e}")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoints": ["/ingest-video", "/query", "/cleanup"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Frontend polls this to check if ingestion is done."""
    status = ingestion_status.get(video_id, "not_started")
    return {"status": status}


@app.post("/ingest-video", status_code=202)
async def start_ingestion_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers the long-running video ingestion process using BackgroundTasks.
    """
    # Validate basic YouTube URL format
    if "youtube.com" not in request.url and "youtu.be" not in request.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL provided.")

    background_tasks.add_task(
        background_ingestion_task, 
        youtube_url=request.url, 
        index_name=PINECONE_INDEX
    )
    
    return {
        "status": "Processing started",
        "message": f"Ingestion for {request.url} has begun. Check server logs for progress."
    }

@app.post("/query")
async def ask_query_endpoint(request: QueryRequest):
    """
    Answers a query using the RAG chain.
    """
    try:
        # Check connectivity
        available_indexes = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX not in available_indexes:
             raise HTTPException(
                status_code=404, 
                detail=f"Index '{PINECONE_INDEX}' not found. Did you ingest any videos yet?"
            )

        # Initialize Chain
        rag_chain = get_rag_chain(PINECONE_INDEX)
        
        # Invoke Chain
        answer = rag_chain.invoke(request.query)
        
        return {
            "query": request.query,
            "answer": answer
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup", response_model=Dict[str, Any])
async def cleanup_system(request: DeleteRequest = DeleteRequest()):
    """
    Cleans up Pinecone index and local transcript files.
    """
    results = {
        "operation": "cleanup",
        "timestamp": time.time(),
        "details": {}
    }
    
    target_index = request.index_name or PINECONE_INDEX
    
    # 1. Delete Pinecone Index
    if request.delete_index:
        try:
            current_indexes = [i.name for i in pc.list_indexes()]
            if target_index in current_indexes:
                pc.delete_index(target_index)
                results["details"]["pinecone"] = "Deleted"
            else:
                results["details"]["pinecone"] = "Not Found (Skipped)"
        except Exception as e:
            results["details"]["pinecone"] = f"Error: {str(e)}"

    # 2. Delete Transcripts Folder
    if request.delete_transcripts:
        transcripts_dir = Path("./transcripts")
        if transcripts_dir.exists():
            try:
                shutil.rmtree(transcripts_dir)
                results["details"]["local_files"] = "Deleted ./transcripts folder"
            except Exception as e:
                results["details"]["local_files"] = f"Error: {str(e)}"
        else:
             results["details"]["local_files"] = "Not Found (Skipped)"

    # 2. Delete temp_workspace Folder
    if request.delete_transcripts:
        transcripts_dir = Path("./temp_workspace")
        if transcripts_dir.exists():
            try:
                shutil.rmtree(transcripts_dir)
                results["details"]["local_files"] = "Deleted ./temp_workspace folder"
            except Exception as e:
                results["details"]["local_files"] = f"Error: {str(e)}"
        else:
             results["details"]["local_files"] = "Not Found (Skipped)"

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)