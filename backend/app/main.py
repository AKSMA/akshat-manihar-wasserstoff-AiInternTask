from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
from config import groq_api_key
from routes import api_router

# Disable parallelism warning for tokenizers (optional, for cleaner logs)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create FastAPI app instance
app = FastAPI()

# Include API routes from routes.py
app.include_router(api_router)

# Run the FastAPI app with Uvicorn server if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)