import os
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Get API keys and URLs from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# This is a secret key you'll create to protect your own API endpoint
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY") 

# --- Security ---
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# --- Pydantic Models for Data Validation ---
class MemoryMetadata(BaseModel):
    collection: str = "episodic_memory" # Default collection
    source: str
    timestamp: str
    tags: list[str] = []

class MemoryInput(BaseModel):
    text: str
    metadata: MemoryMetadata

# --- FastAPI App ---
app = FastAPI(title="Buffy Prime Memory API")

@app.post("/save-memory/", summary="Embed text and save to Qdrant")
def save_memory(memory_input: MemoryInput, secure: str = Depends(get_api_key)):
    """
    This endpoint receives text, generates a Gemini embedding, 
    and stores it in a specified Qdrant collection.
    """
    # 1. Get embedding from Gemini
    try:
        gemini_embed_url = f"https://generativelanugage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
        response = requests.post(gemini_embed_url, json={"model": "models/embedding-001", "content": {"parts": [{"text": memory_input.text}]}})
        response.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
        embedding = response.json()["embedding"]["values"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

    # 2. Save embedding to Qdrant
    try:
        collection_name = memory_input.metadata.collection
        qdrant_points_url = f"{QDRANT_URL}/collections/{collection_name}/points?wait=true"
        
        # We need to upsert points using a unique ID. We'll use the text hash for simplicity.
        import uuid
        point_id = str(uuid.uuid4())

        payload = {
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": memory_input.metadata.dict() # Pass the metadata object
            }]
        }
        
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.put(qdrant_points_url, json=payload, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")

    return {"status": "success", "id": point_id, "collection": collection_name}
