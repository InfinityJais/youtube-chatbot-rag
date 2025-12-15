#!/usr/bin/env python3
"""
Download audio from YouTube, split into WAV chunks (ffmpeg), then transcribe chunks
in parallel (limited concurrency) using OpenAI Whisper (translate -> English).

Usage:
    python download_and_transcribe_parallel.py <youtube_url> [output_folder]
Example:
    python download_and_transcribe_parallel.py "https://youtu.be/abcd" transcripts
"""
import sys
import os
import json
import shutil
import subprocess
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import yt_dlp
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
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


# --- OpenAI client detection (new vs legacy) ---
USE_NEW_CLIENT = False
openai_client = None
legacy_openai = None
# ============================================================
#   Unified Whisper Transcription / Translation API Wrapper
#   Works for BOTH: new OpenAI client and legacy openai package
# ============================================================
try:
    # New OpenAI SDK (2024+)
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    USING_NEW_CLIENT = True

    def transcribe_with_whisper(file_path, model="whisper-1", translate=True):
        """
        translate=True  → return English translation
        translate=False → return original-language transcription
        """
        with open(file_path, "rb") as f:
            if translate:
                # NEW CLIENT → use translations endpoint
                resp = openai_client.audio.translations.create(
                    model=model,
                    file=f
                )
            else:
                # NEW CLIENT → normal transcription (NO 'task' parameter)
                resp = openai_client.audio.transcriptions.create(
                    model=model,
                    file=f
                )

        return resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)

except Exception:
    # Legacy openai package
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    legacy_openai = openai
    USING_NEW_CLIENT = False

    def transcribe_with_whisper(file_path, model="whisper-1", translate=True):
        """
        translate=True  → openai.Audio.translate()
        translate=False → openai.Audio.transcribe()
        """
        with open(file_path, "rb") as f:
            if translate:
                resp = openai.Audio.translate(model=model, file=f)
            else:
                resp = openai.Audio.transcribe(model=model, file=f)

        return resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)

# --- Helpers: download & convert ---
def download_audio(youtube_url: str, out_dir: Path) -> Path:
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
        ext = info.get("ext") or info.get("requested_formats", [{}])[0].get("ext", "webm")
        downloaded = out_dir / f"{video_id}.{ext}"

    wav_path = out_dir / f"{video_id}.wav"
    print(f"[download] Converting {downloaded.name} → {wav_path.name} (16k mono s16)...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(downloaded),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(wav_path)
    ], check=True)
    return wav_path, info

# --- Helpers: audio splitting ---
def get_duration_seconds(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(file_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    duration = None
    if "format" in info and "duration" in info["format"]:
        duration = float(info["format"]["duration"])
    else:
        streams = info.get("streams", [])
        for s in streams:
            if "duration" in s:
                duration = float(s["duration"])
                break
    if duration is None:
        raise RuntimeError("Could not determine duration")
    return duration

# --- Helpers: audio splitting into chunks ---
def split_with_ffmpeg_seek(wav_path: Path, out_dir: Path, chunk_length_s=180, overlap_s=2):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = get_duration_seconds(wav_path)
    print(f"[split] WAV duration: {total:.2f}s")
    step = chunk_length_s - overlap_s
    if step <= 0:
        raise ValueError("chunk_length_s must be > overlap_s")

    chunks = []
    start = 0.0
    idx = 0
    while start < total:
        end = min(start + chunk_length_s, total)
        out_file = out_dir / f"chunk_{idx:03d}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start}",
            "-i", str(wav_path),
            "-t", f"{end - start}",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(out_file)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append((out_file, start, end))
        idx += 1
        start += step
    print(f"[split] Created {len(chunks)} chunks in {out_dir.resolve()}")
    return chunks


# --- Transcription worker (per chunk) ---
def transcribe_chunk_worker(chunk_info, model="whisper-1", max_retries=3, translate=True):
    """
    Robust worker for transcribing one chunk.
    ... (rest of the worker function remains the same)
    """
    chunk_path, start_s, end_s = chunk_info
    chunk_path = Path(chunk_path)
    last_exc = None

    # Quick pre-check
    if not chunk_path.exists():
        err = f"Chunk file not found: {chunk_path}"
        return {"start": start_s, "end": end_s, "text": "", "file": str(chunk_path), "error": err, "retries": 0}

    # Try up to max_retries attempts (attempts are counted from 1..max_retries)
    for attempt in range(1, max_retries + 1):
        try:
            text = transcribe_with_whisper(str(chunk_path), model=model, translate=translate)
            if text is None:
                text = ""
            text = text.strip()
            return {"start": start_s, "end": end_s, "text": text, "file": str(chunk_path), "retries": attempt - 1}
        except Exception as e:
            last_exc = e
            # If this was the last allowed attempt, handle permanent failure
            if attempt == max_retries:
                errmsg = f"Failed chunk {chunk_path.name} after {max_retries} retries. Last error: {e}"
                # try to copy offending chunk for offline inspection (optional)
                try:
                    bad_dir = Path("tmp_chunks")
                    bad_dir.mkdir(exist_ok=True)
                    bad = bad_dir / f"bad_{chunk_path.name}"
                    shutil.copy2(chunk_path, bad)
                    # Return structured error including where the bad copy was saved
                    return {
                        "start": start_s,
                        "end": end_s,
                        "text": "",
                        "file": str(chunk_path),
                        "error": errmsg,
                        "bad_copy": str(bad),
                        "retries": max_retries
                    }
                except Exception as copy_err:
                    # If copying fails, still return the main error
                    return {
                        "start": start_s,
                        "end": end_s,
                        "text": "",
                        "file": str(chunk_path),
                        "error": errmsg + f" (also failed to copy chunk: {copy_err})",
                        "retries": max_retries
                    }
            # not last attempt => exponential backoff then retry
            wait = (2 ** attempt) + random.random()
            time.sleep(wait)

    # Fallback: should not be reached, but return last exception if it does
    return {
        "start": start_s,
        "end": end_s,
        "text": "",
        "file": str(chunk_path),
        "error": str(last_exc),
        "retries": max_retries
    }


def stitch_transcripts(results):
    # results is a list of dicts with start,end,text
    sorted_res = sorted(results, key=lambda r: r["start"])
    lines = []
    for r in sorted_res:
        lines.append(f"[{r['start']:.2f} - {r['end']:.2f}] {r.get('text','')}\n")
    return "".join(lines)


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
        return


    vec = embeddings.embed_query("hello")
    dim = len(vec)
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
def chunk_text(text: str, video_info: Dict, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Use LangChain RecursiveCharacterTextSplitter to produce chunks.
    Returns list of langchain.schema.Document objects with video metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # The split_text method doesn't take metadata, so we split first, then create Documents.
    texts = splitter.split_text(text) 
    
    docs = []
    # Extract common metadata for all chunks from the video info
    common_metadata = {
        "source": video_info.get("webpage_url", "N/A"),
        "video_id": video_info.get("id", "N/A"),
        "title": video_info.get("title", "N/A"),
        "uploader": video_info.get("uploader", "N/A"),
        "upload_date": video_info.get("upload_date", "N/A"),
    }
    
    for i, t in enumerate(texts):
        # We try to infer a rough start time for the chunk by looking at the first timestamp line
        # This is a simplification; a more advanced method would map chunk text back to the transcript results.
        start_time_s = -1 
        try:
            # Look for the timestamp format "[start - end]" at the beginning of the chunk
            time_str = t.split('\n', 1)[0].strip()
            if time_str.startswith('[') and time_str.endswith(']'):
                start_time_s = float(time_str[1:].split(' - ')[0])
        except Exception:
            pass # Ignore if timestamp extraction fails
            
        metadata = common_metadata.copy()
        metadata["chunk"] = i
        if start_time_s >= 0:
             metadata["start_time_s"] = start_time_s
             metadata["url"] = f"{metadata['source']}&t={int(start_time_s)}s" # deep link URL
             
        # Generate a unique ID for the chunk (e.g., videoID-chunkIndex)
        chunk_id = f"{metadata['video_id']}-{i}"
             
        docs.append(Document(
            page_content=t, 
            metadata={**metadata, "id": chunk_id} # Combine metadata and add id
        ))
    return docs

# ------ genrate embedding and upsert documents into pinecone ------
def upsert_documents(index_name: str, docs: List[Document], embeddings) -> Dict:
    """
    Upsert documents into Pinecone in batches.
    Each Document has .page_content and .metadata
    """
    BATCH = 32
    index = pc.Index(index_name)

    # build vectors in batches
    to_upsert = []
    
    print(f"\n[pinecone] Starting upsert of {len(docs)} chunks...")
    
    for i, doc in enumerate(docs):
        # Use the ID from the metadata
        uid = doc.metadata.get("id") or f"doc-{int(time.time())}-{i}"
        
        # NOTE: LangChain's PineconeVectorStore handles the embedding and upsert internally
        # when using .from_documents, but since we're using a manual loop, we need to 
        # generate the vector and prepare the data manually for the Pinecone client.
        
        # We'll use the LangChain PineconeVectorStore for simplicity in a real RAG setup, 
        # but for direct upsert, we need the vector:
        vec = embeddings.embed_query(doc.page_content)
        meta = doc.metadata or {}
        
        # Pinecone upsert format: (id, vector_values, metadata_dict)
        # Note: The 'page_content' text is not stored as part of the Pinecone vector/metadata 
        # (it's implicit in the vector), but is used to generate the embedding. 
        # The LangChain `PineconeVectorStore` stores the text in a metadata field called 'text' 
        # by default, so we'll add it here for consistency if we were to use their retrieval.
        meta['text'] = doc.page_content 
        
        to_upsert.append((uid, vec, meta))

        if len(to_upsert) >= BATCH:
            print(f"  > Upserting batch of {len(to_upsert)}...")
            index.upsert(vectors=to_upsert)
            to_upsert = []
            
    if to_upsert:
        print(f"  > Upserting final batch of {len(to_upsert)}...")
        index.upsert(vectors=to_upsert)
        
    print("[pinecone] Upsert complete.")
    
    return {
        "success": True,
        "chunks_indexed": len(docs),
        "index_name": index_name
    }

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main(youtube_url: str, output_folder: str, max_workers: int = 5):
    """
    Main workflow: Download, Split, Transcribe, Index.
    """
    
    # --- 1. Setup Directories ---
    out_dir = Path(output_folder)
    temp_download_dir = out_dir / "audio_download"
    chunk_dir = out_dir / "audio_chunks"
    
    if out_dir.exists():
        # Clear previous run's output and chunks if folder exists
        shutil.rmtree(out_dir)
    temp_download_dir.mkdir(parents=True)
    
    print(f"🎥 Starting transcription and indexing for: {youtube_url}")
    
    try:
        # --- 2. Download and Convert Audio ---
        wav_file, video_info = download_audio(youtube_url, temp_download_dir)
        print(f"✅ Downloaded and converted to WAV: {wav_file.name}")
        
        # --- 3. Split Audio into Chunks ---
        # Using 3-minute chunks with 2-second overlap
        chunks = split_with_ffmpeg_seek(wav_file, chunk_dir, chunk_length_s=180, overlap_s=2)
        
        # --- 4. Parallel Transcription ---
        print(f"\n🎧 Starting parallel transcription of {len(chunks)} chunks (max workers: {max_workers})...")
        all_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the chunks to the transcription worker function
            future_to_chunk = {
                executor.submit(transcribe_chunk_worker, chunk_info): chunk_info 
                for chunk_info in chunks
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_chunk)):
                chunk_info = future_to_chunk[future]
                chunk_file, start_s, end_s = chunk_info
                
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    status = "✅ Success"
                    if result.get("error"):
                        status = f"❌ Error (retries: {result.get('retries', 0)})"
                        print(f"[{i+1}/{len(chunks)}] {status} for {chunk_file.name}")
                    else:
                        print(f"[{i+1}/{len(chunks)}] {status} (retries: {result.get('retries', 0)}) - Text: {result['text'][:50]}...")
                        
                except Exception as exc:
                    print(f"[{i+1}/{len(chunks)}] 💥 UNCAUGHT EXCEPTION processing {chunk_file.name}: {exc}")
                    all_results.append({
                        "start": start_s, "end": end_s, "text": "", "file": str(chunk_file), 
                        "error": str(exc), "retries": 0
                    })
                    
        total_time = time.time() - start_time
        print(f"\n⏱️ Transcription finished. Total time: {total_time:.2f} seconds.")
        
        # --- 5. Stitch Transcripts and Save ---
        full_transcript = stitch_transcripts(all_results)
        transcript_file = out_dir / "full_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"📝 Full transcript saved to: {transcript_file.resolve()}")
        
        # --- 6. Prepare for RAG Indexing ---
        
        # Instantiate Embeddings model (used for Pinecone dimension check and upsert)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
        
        # Ensure Pinecone index exists
        init_pinecone(PINECONE_INDEX, embeddings)
        
        # Chunk the stitched transcript for indexing
        # Note: We pass the video_info to enrich the metadata of the chunks
        rag_documents = chunk_text(
            text=full_transcript, 
            video_info=video_info, 
            chunk_size=1000, 
            chunk_overlap=200
        )
        print(f"✂️ Split transcript into {len(rag_documents)} RAG chunks.")
        
        # --- 7. Upsert to Pinecone ---
        upsert_result = upsert_documents(PINECONE_INDEX, rag_documents, embeddings)
        print(f"💾 Indexing complete: {upsert_result.get('chunks_indexed')} chunks added to index '{PINECONE_INDEX}'.")

        # --- 8. RAG Setup for Querying (Optional/Next Step) ---
        print("\n🧠 Setting up LangChain RAG pipeline for querying...")
        
        # Create a retriever from the vector store
        # NOTE: PineconeVectorStore requires the index to already have been created 
        # (which init_pinecone ensures)
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX, 
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        retriever = vectorstore.as_retriever()
        
        # Define the LLM and the Prompt Template
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        template = """You are an expert Q&A assistant for YouTube video transcripts.
        Use the following retrieved context, which includes timestamps and source URLs, to answer the question.
        If you cannot find the answer in the provided context, state that clearly and do not make up an answer.
        Always cite the source's title, video ID, and the most relevant timestamp/URL from the context chunks.

        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
        rag_prompt = PromptTemplate.from_template(template)
        
        # Build the RAG Chain
        def format_docs(docs):
            # Format documents to be clean and readable for the LLM
            return "\n\n---\n\n".join([f"Source Title: {doc.metadata.get('title')}\nChunk Start: {doc.metadata.get('start_time_s')}s\nSource URL: {doc.metadata.get('url')}\nContent: {doc.page_content}" for doc in docs])

        rag_chain = (
            RunnableParallel(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
            )
            | rag_prompt
            | llm
            | str
        )
        
        print("💡 RAG Chain is ready to use! Example Query:")
        query = "What was the main topic discussed, and what key point was made around the 180-second mark?"
        print(f"\nQ: {query}")
        
        # Note: We execute the chain here as a final step to show functionality.
        # This will incur a small cost (retrieval + LLM call).
        print("Awaiting LLM response...")
        llm_response = rag_chain.invoke(query)
        
        print("\n--- LLM Response ---")
        print(llm_response)
        print("--------------------")

    except Exception as e:
        print(f"\n🚨 A critical error occurred during the process: {e}")
        # traceback.print_exc()
    finally:
        # Clean up temporary download files
        if temp_download_dir.exists():
             shutil.rmtree(temp_download_dir)
             print(f"\n🧹 Cleaned up temporary download directory: {temp_download_dir}")

# ---------------------------
# Script Execution
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    youtube_url = sys.argv[1]
    # Use "transcripts" as default output folder if not provided
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "transcripts" 
    
    # Run the main process
    main(youtube_url, output_folder)