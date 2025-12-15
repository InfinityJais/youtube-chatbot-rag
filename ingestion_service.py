import os
import json
import shutil
import subprocess
import time
import random
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Union
import logging
logger = logging.getLogger(__name__)

import yt_dlp
from dotenv import load_dotenv

# Add these imports at the top of ingestion_service.py
from config import embeddings, pc, PINECONE_INDEX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document



# NOTE: load_dotenv is assumed to be run by the main FastAPI app, but kept here for local testing potential
# load_dotenv() 

# --- OpenAI Client Detection (V3 vs Legacy) ---
# This block initializes the client and defines the unified wrapper function
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def transcribe_with_whisper(file_path, model="whisper-1", translate=True):
        """Unified wrapper for OpenAI Whisper API call (New SDK)."""
        with open(file_path, "rb") as f:
            if translate:
                # NEW CLIENT → use translations endpoint
                resp = openai_client.audio.translations.create(model=model, file=f)
            else:
                # NEW CLIENT → normal transcription
                resp = openai_client.audio.transcriptions.create(model=model, file=f)
            return resp.text

except Exception:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    legacy_openai = openai
    
    def transcribe_with_whisper(file_path, model="whisper-1", translate=True):
        """Unified wrapper for OpenAI Whisper API call (Legacy SDK)."""
        with open(file_path, "rb") as f:
            if translate:
                resp = legacy_openai.Audio.translate(model=model, file=f)
            else:
                resp = legacy_openai.Audio.transcribe(model=model, file=f)
        return resp.get("text")


# --- I/O Helpers ---

def download_audio(youtube_url: str, out_dir: Path) -> Tuple[Path, Dict[str, Union[str, int]]]:
    """Downloads audio, converts to 16k mono WAV, and returns path/info."""
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
        # Find the actual downloaded file (yt-dlp may use different extensions)
        downloaded = next(out_dir.glob(f"{video_id}.*"), None)
        if downloaded is None:
            raise FileNotFoundError(f"yt-dlp failed to find the downloaded file for {video_id}.")
        
    wav_path = out_dir / f"{video_id}.wav"
    
    # FFmpeg conversion to 16k mono s16 (Synchronous)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(downloaded),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(downloaded) # Clean up original downloaded file
    return wav_path, info

def get_duration_seconds(file_path: Path) -> float:
    """Uses ffprobe to get audio duration."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration:stream=duration", 
           "-of", "json", str(file_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    
    duration = float(info.get("streams", [{}])[0].get("duration", 
                     info.get("format", {}).get("duration", 0)))
    
    if duration <= 0:
        raise RuntimeError("Could not determine audio duration using ffprobe.")
    return duration

def split_with_ffmpeg_seek(wav_path: Path, out_dir: Path, chunk_length_s=180, overlap_s=2):
    """Splits WAV into chunks using ffmpeg."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = get_duration_seconds(wav_path)
    step = chunk_length_s - overlap_s
    
    if step <= 0:
        raise ValueError("chunk_length_s must be > overlap_s")

    chunks = []
    start = 0.0
    idx = 0
    while start < total:
        end = min(start + chunk_length_s, total)
        out_file = out_dir / f"chunk_{idx:03d}.wav"
        
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start}", "-i", str(wav_path), "-t", f"{end - start}",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        chunks.append((out_file, start, end))
        idx += 1
        start += step
    
    return chunks

# Add this helper function for chunking text
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
            "source": str(source_path),
            "chunk_index": i,
            "total_chunks": len(chunks),
            **(metadata or {})
        }
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents


def transcribe_chunk_worker(chunk_info, model="whisper-1", max_retries=3, translate=True):
    """Robust worker for transcribing one chunk with retries."""
    chunk_path, start_s, end_s = chunk_info
    chunk_path = Path(chunk_path)
    attempt = 0

    if not chunk_path.exists():
        return {"start": start_s, "end": end_s, "text": "", "file": str(chunk_path), "error": "Chunk file not found."}

    while attempt <= max_retries:
        try:
            text = transcribe_with_whisper(str(chunk_path), model=model, translate=translate)
            return {"start": start_s, "end": end_s, "text": (text or "").strip(), "file": str(chunk_path)}
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                errmsg = f"Failed chunk {chunk_path.name} after {max_retries} retries. Last error: {e}"
                return {"start": start_s, "end": end_s, "text": f"**TRANSCRIPTION ERROR**", "file": str(chunk_path), "error": errmsg}
            time.sleep((2 ** attempt) + random.random())
    return {"start": start_s, "end": end_s, "text": f"**TRANSCRIPTION ERROR**"}

def stitch_transcripts(results, title, output_folder) -> Path:
    """Stitches results and saves the final transcript file."""
    results.sort(key=lambda r: r["start"])
    stitched_lines = [r["text"] for r in results if r["text"] and not r["text"].startswith('**TRANSCRIPTION ERROR**')]
    
    stitched = "\n".join(stitched_lines)
    
    # Build safe filename
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()[:200]
    out_txt = output_folder / f"{safe}_translated.txt"
    out_txt.parent.mkdir(exist_ok=True)
    out_txt.write_text(stitched, encoding="utf-8")
    
    return out_txt


# ----------------------------------------------------------------------
# THE CORE CALLABLE API FUNCTION
# ----------------------------------------------------------------------

# Update the main function signature and logic
def run_transcription_pipeline_core(
    youtube_url: str, 
    pinecone_index_name: str,  # Changed from output_folder
    output_base_dir: Path = Path("transcripts")  # Add default
) -> dict:
    """
    Complete pipeline: Download → Transcribe → Store in Pinecone → Save file.
    Returns: Dictionary with results
    """
    import tempfile
    from pathlib import Path
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    CHUNKS_DIR = temp_dir / "chunks"
    
    try:
        # 1. Download and transcribe
        logger.info(f"[TRANSCRIPTION] Starting for: {youtube_url}")
        wav_path, info = download_audio(youtube_url, temp_dir)
        
        # 2. Split audio
        chunks = split_with_ffmpeg_seek(wav_path, CHUNKS_DIR)
        
        # 3. Transcribe in parallel
        results = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(transcribe_chunk_worker, c) for c in chunks]
            for fut in as_completed(futures):
                results.append(fut.result())
        
        # 4. Stitch full transcript
        results.sort(key=lambda r: r["start"])
        full_text = " ".join([r["text"] for r in results if r.get("text")])
        
        video_id = info.get("id", "unknown")
        video_title = info.get("title") or "YouTube Video"
        
        # 5. Create and store embeddings in Pinecone
        metadata = {
            "video_id": video_id,
            "video_title": video_title,
            "source_url": youtube_url,
            "processed_at": time.time()
        }
        
        documents = _chunk_and_enrich(full_text, Path(f"youtube:{video_id}"), metadata)
        
        # Store in Pinecone
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=pinecone_index_name
        )
        
        logger.info(f"[TRANSCRIPTION] Stored {len(documents)} chunks in Pinecone index '{pinecone_index_name}'")
        
        # 6. Save transcript file (optional)
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (" ", "_", "-")).rstrip()[:200]
        transcript_path = output_base_dir / f"{safe_title}_translated.txt"
        transcript_path.parent.mkdir(exist_ok=True)
        transcript_path.write_text(full_text, encoding="utf-8")
        
        return {
            "success": True,
            "video_id": video_id,
            "video_title": video_title,
            "chunks_indexed": len(documents),
            "transcript_path": str(transcript_path),
            "pinecone_index": pinecone_index_name
        }
        
    except Exception as e:
        logger.error(f"[TRANSCRIPTION] FAILED: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "video_url": youtube_url
        }
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)