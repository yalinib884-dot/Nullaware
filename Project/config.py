"""
config.py - Central configuration for NulAware AI
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Storage
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
PROFILE_JSON_DIR = BASE_DIR / os.getenv("PROFILE_JSON_DIR", "storage/profile_json")
REPORTS_DIR = BASE_DIR / os.getenv("REPORTS_DIR", "storage/reports")
CHARTS_DIR = BASE_DIR / os.getenv("CHARTS_DIR", "storage/charts")
CHROMA_PERSIST_DIR = BASE_DIR / "storage/chromadb"
SQLITE_DB_PATH = BASE_DIR / os.getenv("SQLITE_DB_PATH", "storage/nulaware.db")

# Ensure dirs exist
# Ensure dirs exist
for d in [
    UPLOAD_DIR,
    PROFILE_JSON_DIR,
    REPORTS_DIR,
    CHARTS_DIR,
    CHROMA_PERSIST_DIR,
    SQLITE_DB_PATH.parent,
]:
    d.mkdir(parents=True, exist_ok=True)

# Gemini API keys
GEMINI_API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
        os.getenv("GEMINI_API_KEY_4"),
        os.getenv("GEMINI_API_KEY_5"),
    ] if k
]

# Model
GEMINI_MODEL = "gemini-2.5-flash"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))

# ChromaDB collection
CHROMA_COLLECTION = "nulaware_chunks"

