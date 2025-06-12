from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory where uploaded files are stored
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# Directory or path for the FAISS vector index
INDEX_DIR = "backend/faiss_index"

# API key for Groq LLM, loaded from environment variable
groq_api_key = os.environ['GROQ_API_KEY']