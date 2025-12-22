import os
import json
import traceback
import shutil
import subprocess
import time
import random
import logging
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional, Union

import yt_dlp
from dotenv import load_dotenv

# --- Configuration Imports ---
# Wrap in try/except so script doesn't crash if config.py is missing during testing
try:
    from config import embeddings, pc, PINECONE_INDEX, out_base
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_pinecone import PineconeVectorStore
    from langchain_core.documents import Document
except ImportError:
    # Placeholder for running standalone
    pass

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- OpenAI Client Setup (Robust V1/Legacy Support) ---
try:
    from openai import OpenAI
    # Initialize client once
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def transcribe_with_whisper(file_path: str, model="whisper-1", translate=True) -> str:
        """Unified wrapper for OpenAI Whisper API call (New SDK >= 1.0.0)."""
        with open(file_path, "rb") as f:
            if translate:
                resp = openai_client.audio.translations.create(model=model, file=f)
            else:
                resp = openai_client.audio.transcriptions.create(model=model, file=f)
            # V1 returns an object, .text is the attribute
            return resp.text

except ImportError:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def transcribe_with_whisper(file_path: str, model="whisper-1", translate=True) -> str:
        """Unified wrapper for OpenAI Whisper API call (Legacy SDK < 1.0.0)."""
        with open(file_path, "rb") as f:
            if translate:
                resp = openai.Audio.translate(model=model, file=f)
            else:
                resp = openai.Audio.transcribe(model=model, file=f)
        return resp.get("text", "")


# --- Core Functions ---

def download_audio(youtube_url: str, out_dir: Path) -> Tuple[Path, Dict]:
    """Downloads audio using yt-dlp and returns the path to the converted WAV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a safe temp filename template
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    
    wav_path = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Fetching info for: {youtube_url}")
            info = ydl.extract_info(youtube_url, download=True)
            
            # FIXED: Don't guess the filename using glob. Ask yt-dlp what it wrote.
            downloaded_path = Path(ydl.prepare_filename(info))
            
            # Handle case where yt-dlp converts format post-download (rare with these opts but possible)
            if not downloaded_path.exists():
                # Fallback: look for likely candidates if prepare_filename is slightly off due to post-processing
                video_id = info.get('id')
                candidates = list(out_dir.glob(f"{video_id}.*"))
                if candidates:
                    downloaded_path = candidates[0]
                else:
                    raise FileNotFoundError(f"Could not locate downloaded file for ID: {video_id}")

            video_id = info.get("id")
            wav_path = out_dir / f"{video_id}.wav"
            
            logger.info(f"Converting {downloaded_path.name} to WAV...")
            
            # FFmpeg conversion to 16k mono s16
            subprocess.run([
                "ffmpeg", "-y", "-i", str(downloaded_path),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(wav_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            # Optional: Remove original file to save space
            #if downloaded_path.exists() and downloaded_path != wav_path:
            #    os.remove(downloaded_path)

            return wav_path, info

    except Exception as e:
        logger.error(f"Error in download_audio: {e}")
        raise

def get_duration_seconds(file_path: Path) -> float:
    """Uses ffprobe to get audio duration safely."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", 
        "-show_entries", "format=duration:stream=duration", 
        "-of", "json", str(file_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        
        # Try stream duration first, then format duration
        duration_str = None
        if "streams" in data and data["streams"]:
            duration_str = data["streams"][0].get("duration")
        
        if not duration_str and "format" in data:
            duration_str = data["format"].get("duration")
            
        if duration_str:
            return float(duration_str)
        else:
            raise ValueError("No duration found in ffprobe output")
            
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to get duration for {file_path}: {e}")
        raise RuntimeError(f"Could not determine audio duration: {e}")

def split_with_ffmpeg_seek(wav_path: Path, out_dir: Path, chunk_length_s=180, overlap_s=2) -> List[Tuple[Path, float, float]]:
    """Splits WAV into chunks using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_duration = get_duration_seconds(wav_path)
    step = chunk_length_s - overlap_s
    
    if step <= 0:
        raise ValueError("chunk_length_s must be > overlap_s")

    chunks_info = []
    start = 0.0
    idx = 0
    
    logger.info(f"Splitting audio (Total: {total_duration:.2f}s) into chunks...")
    
    while start < total_duration:
        end = min(start + chunk_length_s, total_duration)
        chunk_filename = f"chunk_{idx:03d}.wav"
        out_file = out_dir / chunk_filename
        
        # We process strictly to avoid drift, using -ss before -i for speed
        duration = end - start
        
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(wav_path), 
            "-t", f"{duration:.2f}",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        chunks_info.append((out_file, start, end))
        
        if end >= total_duration:
            break
            
        idx += 1
        start += step
    
    return chunks_info

def transcribe_chunk_worker(chunk_info: Tuple[Path, float, float], model="whisper-1", max_retries=3, translate=True) -> Dict:
    """Robust worker for transcribing one chunk with retries."""
    chunk_path, start_s, end_s = chunk_info
    attempt = 0

    if not chunk_path.exists():
        return {"start": start_s, "end": end_s, "text": "", "error": "File missing"}

    while attempt <= max_retries:
        try:
            # logger.debug(f"Transcribing chunk {chunk_path.name} (Attempt {attempt+1})")
            text = transcribe_with_whisper(str(chunk_path), model=model, translate=translate)
            return {"start": start_s, "end": end_s, "text": (text or "").strip(), "file": str(chunk_path)}
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Failed chunk {chunk_path.name}: {e}")
                return {
                    "start": start_s, "end": end_s, 
                    "text": "**TRANSCRIPTION ERROR**", 
                    "error": str(e)
                }
            # Exponential backoff
            time.sleep((1.5 ** attempt) + random.uniform(0.1, 1.0))
    
    return {"start": start_s, "end": end_s, "text": "**TRANSCRIPTION ERROR**"}

def stitch_transcripts(results: List[Dict], title: str, output_folder: Path) -> Path:
    """Stitches results and saves the final transcript file."""
    # Sort by start time to ensure order
    results.sort(key=lambda r: r["start"])
    
    stitched_lines = []
    for r in results:
        text = r.get("text", "")
        if text and not text.startswith('**TRANSCRIPTION ERROR**'):
            stitched_lines.append(text)
    
    full_text = "\n".join(stitched_lines)
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()[:100]
    out_file = output_folder / f"{safe_title}_transcript.txt"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    out_file.write_text(full_text, encoding="utf-8")
    
    logger.info(f"Transcript saved to: {out_file}")
    return out_file

# --- Orchestrator (The Missing Piece) ---

def transcribe_video_pipeline(youtube_url: str, working_dir="./temp_workspace") -> Optional[Path]:
    """
    Main pipeline function: Download -> Split -> Transcribe (Parallel) -> Stitch
    """
    session_id = str(uuid.uuid4())[:8]
    work_path = Path(working_dir) / session_id
    raw_audio_dir = work_path / "raw"
    chunks_dir = work_path / "chunks"
    
    try:
        # 1. Download
        wav_path, info = download_audio(youtube_url, raw_audio_dir)
        video_title = info.get("title", "Unknown Video")
        
        # 2. Split
        # 10 minute chunks usually work well for Whisper API limits (25MB)
        chunks = split_with_ffmpeg_seek(wav_path, chunks_dir, chunk_length_s=300, overlap_s=5)
        
        # 3. Parallel Transcription
        results = []
        max_workers = min(len(chunks), 10) # Don't exceed rate limits excessively
        
        logger.info(f"Starting parallel transcription with {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures to chunks
            future_to_chunk = {
                executor.submit(transcribe_chunk_worker, chunk): chunk 
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    logger.error(f"Chunk processing generated an exception: {exc}")

        # 4. Stitch
        final_transcript_path = stitch_transcripts(results, video_title, Path("./transcripts"))
        
        return final_transcript_path

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return None
        
    finally:
        # cleanup temp files
        if work_path.exists():
            logger.info(f"Cleaning up workspace: {work_path}")
            shutil.rmtree(work_path)

# --- Entry Point ---

if __name__ == "__main__":
    # Example Usage
    test_url = "https://www.youtube.com/watch?v=vkhjO7fc78g" 
    
    # Ensure you set your API Key in .env or environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables!")
    
    print(f"Processing: {test_url}")
    result_file = transcribe_video_pipeline(test_url)
    if result_file:
        print(f"Success! Saved at: {result_file}")