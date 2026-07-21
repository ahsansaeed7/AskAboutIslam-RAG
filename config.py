import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Make all paths absolute, anchored to this config.py file's location (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Chroma settings
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "vector_store", "chroma_db")
CHROMA_COLLECTION_NAME = "islam_wikipedia"

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# Switch between "ollama" (local dev) and "groq" (deployed)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # defaults to ollama for local dev