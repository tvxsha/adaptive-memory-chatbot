"""Central config, loaded from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.db")

# The memory categories described in the project objectives
CATEGORIES = [
    "fact",
    "preference",
    "goal",
    "decision",
    "completed_task",
    "recent_chat",
]

if not GROQ_API_KEY:
    print(
        "[memory_engine.config] WARNING: GROQ_API_KEY is not set. "
        "Copy .env.example to .env and add your key before running the server."
    )

if not HF_TOKEN:
    print(
        "[memory_engine.config] WARNING: HF_TOKEN is not set. "
        "Embeddings (via the Hugging Face Inference API) will fail without it. "
        "Get a free token at https://huggingface.co/settings/tokens"
    )
