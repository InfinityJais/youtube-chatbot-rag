# main_api.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from config import PINECONE_INDEX, pc
from rag_core import get_rag_chain
from ingestion_service import run_transcription_pipeline_core
from typing import Optional, List
import logging
import shutil
from typing import Any, Dict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

class DeleteRequest(BaseModel):
    delete_index: bool = True
    delete_transcripts: bool = True


app = FastAPI(
    title="YouTube Transcript RAG API",
    description="RAG system for YouTube video Q&A",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://127.0.0.1:8000/docs
    redoc_url="/redoc"  # ReDoc at http://127.0.0.1:8000/redoc
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "YouTube Transcript RAG API",
        "status": "running",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "health": "/health",
            "ingest": "/api/ingest",
            "query": "/api/query",
            "search": "/api/search"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "youtube-rag-api"}

# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working!"}

# ----------------------------------------------------------------------
# 1. INGESTION ENDPOINT (Background Task)
# ----------------------------------------------------------------------
@app.post("/ingest-video", status_code=202)
async def start_ingestion_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers the long-running video ingestion process using BackgroundTasks.
    """
    url = request.url
    
    background_tasks.add_task(
    run_transcription_pipeline_core, 
    youtube_url=url, 
    pinecone_index_name=PINECONE_INDEX, 
    output_base_dir=Path("transcripts")   
    )
    
    logger.info(f"Ingestion started for URL: {url}")  

    return {
        "status": "Processing started",
        "message": f"Ingestion for {url} has begun in the background. Check server logs for status."
    }

# ----------------------------------------------------------------------
# 2. QUERY ENDPOINT (Short Task)
# ----------------------------------------------------------------------
@app.post("/query")
async def ask_query_endpoint(request: QueryRequest):
    """
    Answers a query using the RAG chain.
    """
    try:
        # Check if the index is ready (optional, but good practice)
        if PINECONE_INDEX not in pc.list_indexes().names():
             raise HTTPException(
                status_code=404, 
                detail=f"Vector Index '{PINECONE_INDEX}' not found. Please wait for ingestion to complete."
            )

        # Initialize and invoke the LCEL RAG chain
        rag_chain = get_rag_chain(PINECONE_INDEX)
        answer = rag_chain.invoke(request.query)
        
        return {
            "query": request.query,
            "answer": answer
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"RAG QUERY ERROR: {e}")
        raise HTTPException(
            status_code=503, 
            detail="RAG query failed due to internal service error."
        )


# Add this to main_api.py after your other endpoints

# ----------------------------------------------------------------------
# 3. DELETE ENDPOINT (Cleanup/Reset)
# ----------------------------------------------------------------------
@app.delete("/cleanup", response_model=Dict[str, Any], summary="Cleans up Pinecone index and local transcripts.")

async def cleanup_system(request: Optional[DeleteRequest] = None):
    """
    Handles system cleanup operations based on the request parameters.
    - If `delete_index` is True, it deletes the specified Pinecone index (or default).
    - If `delete_transcripts` is True, it deletes the local 'transcripts' folder.
    """
    results = {
        "operation": "system_cleanup",
        "success": True,
        "timestamp": time.time(),
        "details": {}
    }
    
    # Use provided parameters or defaults
    if request is None:
        request = DeleteRequest()  # Use defaults (all True)
    
    # Determine which index to target, defaulting to the global config
    target_index = request.index_name or PINECONE_INDEX
    
    logger.info(f"[CLEANUP] Starting cleanup with parameters: {request.model_dump()}")
    
    try:
        # 1. Delete Pinecone Index
        if request.delete_index:
            try:
                index_names = pc.list_indexes().names()
                if target_index in index_names:
                    pc.delete_index(target_index)
                    results["details"]["pinecone_index"] = {
                        "deleted": target_index,
                        "status": "success"
                    }
                    logger.info(f"[CLEANUP] Deleted Pinecone index: {target_index}")
                else:
                    results["details"]["pinecone_index"] = {
                        "deleted": target_index,
                        "status": "not_found"
                    }
                    logger.info(f"[CLEANUP] Pinecone index not found: {target_index}")
            except Exception as e:
                results["details"]["pinecone_index"] = {
                    "deleted": target_index,
                    "status": "error",
                    "error": str(e)
                }
                results["success"] = False
                logger.error(f"[CLEANUP] Failed to delete index {target_index}: {e}")
        
        # 2. Delete Transcript Files (and the containing folder)
        if request.delete_transcripts:
            try:
                # The folder name used in the previous script's default main()
                transcripts_dir = Path("transcripts") 
                if transcripts_dir.exists() and transcripts_dir.is_dir():
                    # Count files before deletion (just the top-level folder count is enough for logs)
                    file_count = sum(1 for _ in transcripts_dir.rglob("*") if _.is_file())
                    
                    # Delete directory and all contents
                    shutil.rmtree(transcripts_dir)
                    
                    results["details"]["transcripts"] = {
                        "deleted": str(transcripts_dir),
                        "files_deleted": file_count,
                        "status": "success"
                    }
                    logger.info(f"[CLEANUP] Deleted directory '{transcripts_dir}' containing {file_count} files.")
                else:
                    results["details"]["transcripts"] = {
                        "deleted": str(transcripts_dir),
                        "status": "not_found"
                    }
                    logger.info(f"[CLEANUP] Local transcripts directory not found: {transcripts_dir}")
            except Exception as e:
                results["details"]["transcripts"] = {
                    "status": "error",
                    "error": str(e)
                }
                results["success"] = False
                logger.error(f"[CLEANUP] Failed to delete transcripts: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"[CLEANUP] System cleanup failed: {e}")
        # The HTTPException is outside the specific cleanup logic for Pinecone/transcripts, 
        # handling severe errors (like Pinecone client initialization failure, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )
# --- How to Run ---
# uvicorn main_api:app --reload