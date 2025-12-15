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
from dotenv import load_dotenv

load_dotenv()

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
    - chunk_info: (Path|str, start_s, end_s)
    - Uses transcribe_with_whisper(...) wrapper (must exist in script)
    - Retries on exception with exponential backoff.
    - Returns a dict: {start, end, text, file, error?}
    """
    chunk_path, start_s, end_s = chunk_info
    chunk_path = Path(chunk_path)
    attempt = 0
    last_exc = None

    # Quick pre-check
    if not chunk_path.exists():
        err = f"Chunk file not found: {chunk_path}"
        print(f"[worker] {err}")
        return {"start": start_s, "end": end_s, "text": "", "file": str(chunk_path), "error": err}

    print(f"[worker] Starting transcription for {chunk_path.name} (size={chunk_path.stat().st_size} bytes)")

    while attempt <= max_retries:
        try:
            # Call the unified wrapper (it handles new vs legacy client and translate arg)
            text = transcribe_with_whisper(str(chunk_path), model=model, translate=translate)
            if text is None:
                text = ""
            text = text.strip()
            return {"start": start_s, "end": end_s, "text": text, "file": str(chunk_path)}
        except Exception as e:
            last_exc = e
            attempt += 1
            # print short traceback for debugging
            print(f"[worker] Error on {chunk_path.name} (attempt {attempt}/{max_retries}): {e}")
            traceback.print_exc(limit=1)
            if attempt > max_retries:
                errmsg = f"Failed chunk {chunk_path.name} after {max_retries} retries. Last error: {e}"
                print(f"[worker] {errmsg}")
                # copy offending chunk for offline inspection (optional)
                try:
                    bad = Path("tmp_chunks") / f"bad_{chunk_path.name}"
                    shutil.copy2(chunk_path, bad)
                    print(f"[worker] Saved copy of bad chunk to: {bad}")
                except Exception as copy_err:
                    print(f"[worker] Could not save bad chunk: {copy_err}")
                return {"start": start_s, "end": end_s, "text": "", "file": str(chunk_path), "error": errmsg}
            # exponential backoff
            wait = (2 ** attempt) + random.random()
            print(f"[worker] Retrying in {wait:.1f}s...")
            time.sleep(wait)

    # should never reach here
    return {"start": start_s, "end": end_s, "text": "", "file": str(chunk_path), "error": str(last_exc)}


def stitch_transcripts(results):
    # results is a list of dicts with start,end,text
    sorted_res = sorted(results, key=lambda r: r["start"])
    lines = []
    for r in sorted_res:
        lines.append(f"[{r['start']:.2f} - {r['end']:.2f}] {r.get('text','')}\n")
    return "".join(lines)

# --- Main flow ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python download_and_transcribe_parallel.py <youtube_url> [output_folder]")
        sys.exit(1)

    youtube_url = sys.argv[1]
    output_folder = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("transcripts")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Optional config (tweak here or extend as CLI args)
    chunk_length_s = 180
    overlap_s = 2
    concurrency = 3            # number of parallel uploads
    max_retries = 3

    tmp_audio = Path("tmp_yt")
    if tmp_audio.exists():
        shutil.rmtree(tmp_audio)
    tmp_audio.mkdir(parents=True, exist_ok=True)

    print("[main] Downloading audio...")
    wav_path, info = download_audio(youtube_url, tmp_audio)

    print("[main] Splitting audio into chunks...")
    chunks = split_with_ffmpeg_seek(wav_path, Path("tmp_chunks"), chunk_length_s=chunk_length_s, overlap_s=overlap_s)

    # quick sanity
    if not chunks:
        print("[main] No chunks created — aborting")
        sys.exit(2)

    print(f"[main] Transcribing {len(chunks)} chunks with concurrency={concurrency} ...")
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_chunk = {ex.submit(transcribe_chunk_worker, c, "whisper-1", max_retries): c for c in chunks}
        for fut in as_completed(future_to_chunk):
            chunk_info = future_to_chunk[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"[main] Completed {Path(res['file']).name}  ({res['start']:.1f}-{res['end']:.1f})")
            except Exception as e:
                print(f"[main] Unexpected error for chunk {chunk_info}: {e}")
                results.append({"start": chunk_info[1], "end": chunk_info[2], "text": "", "file": str(chunk_info[0]), "error": str(e)})

    # Stitch results in order
    stitched = stitch_transcripts(results)

    # Build output file name from video title
    title = info.get("title") or info.get("id") or "youtube_transcript"
    safe = "".join(c for c in title if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()[:200]
    out_txt = output_folder / f"{safe}_translated.txt"
    out_txt.write_text(stitched, encoding="utf-8")
    print(f"[main] Final stitched transcript saved to: {out_txt.resolve()}")

    # Cleanup (remove temp audio and chunks)
    shutil.rmtree(tmp_audio, ignore_errors=True)
    shutil.rmtree(Path("tmp_chunks"), ignore_errors=True)

if __name__ == "__main__":
    main()
