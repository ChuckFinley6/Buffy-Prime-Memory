import os
import requests
import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY") 

# --- Security ---
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# --- Pydantic Models for Data Validation ---
class MemoryMetadata(BaseModel):
    collection: str = "episodic_memory"
    source: str
    timestamp: str
    tags: list[str] = []

class MemoryInput(BaseModel):
    text: str
    metadata: MemoryMetadata

class SearchInput(BaseModel):
    query: str
    collection: str = "episodic_memory"
    limit: int = 3

# --- FastAPI App ---
app = FastAPI(title="Buffy Prime Memory API")

# --- Helper Function for Gemini Embedding ---
def get_gemini_embedding(text: str):
    try:
        gemini_embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
        response = requests.post(gemini_embed_url, json={"model": "models/embedding-001", "content": {"parts": [{"text": text}]}})
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

# --- API Endpoints ---
@app.post("/save-memory/", summary="Embed text and save to Qdrant")
def save_memory(memory_input: MemoryInput, secure: str = Depends(get_api_key)):
    embedding = get_gemini_embedding(memory_input.text)
    
    try:
        collection_name = memory_input.metadata.collection
        qdrant_points_url = f"{QDRANT_URL}/collections/{collection_name}/points?wait=true"
        point_id = str(uuid.uuid4())

        payload = {
            "points": [{ "id": point_id, "vector": embedding, "payload": memory_input.metadata.dict() }]
        }
        
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.put(qdrant_points_url, json=payload, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")

    return {"status": "success", "id": point_id, "collection": collection_name}

@app.post("/search-memory/", summary="Search for relevant memories in Qdrant")
def search_memory(search_input: SearchInput, secure: str = Depends(get_api_key)):
    query_embedding = get_gemini_embedding(search_input.query)

    try:
        collection_name = search_input.collection
        qdrant_search_url = f"{QDRANT_URL}/collections/{collection_name}/points/search"
        
        payload = {
            "vector": query_embedding,
            "limit": search_input.limit,
            "with_payload": True # Include the metadata in the results
        }
        
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.post(qdrant_search_url, json=payload, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")
        
    return response.json()
