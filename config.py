import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "youtube-rag-index")
TEMP_DIR = Path("transcript/audio_chunks")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not loaded. Check your .env file")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not loaded. Check your .env file")


from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)