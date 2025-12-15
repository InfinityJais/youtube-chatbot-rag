import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables once at the start
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "youtube-rag-index")
TEMP_DIR = Path("tmp_video_files")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("API keys not loaded. Check your .env file.")

# --- Clients ---
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Initialize Global Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Transcription Client ---
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)