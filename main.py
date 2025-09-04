# main.py (Correct API Server Code for Render)
import os
import requests
import uuid
import base64
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
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

# --- Pydantic Models ---
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

# --- Helper Functions ---
def get_gemini_embedding(text: str):
    """Calls Gemini to get a text embedding."""
    try:
        gemini_embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
        response = requests.post(gemini_embed_url, json={"model": "models/embedding-001", "content": {"parts": [{"text": text}]}})
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini Embedding API: {e}")

# --- API Endpoints ---
@app.post("/save-memory/", summary="Embed text and save to Qdrant")
def save_memory(memory_input: MemoryInput, secure: str = Depends(get_api_key)):
    embedding = get_gemini_embedding(memory_input.text)
    try:
        collection_name = memory_input.metadata.collection
        qdrant_points_url = f"{QDRANT_URL}/collections/{collection_name}/points"
        point_id = str(uuid.uuid4())
        payload = {
            "points": [{"id": point_id, "vector": embedding, "payload": memory_input.metadata.dict() | {"text": memory_input.text}}]
        }
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
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
            "with_payload": True
        }
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        response = requests.post(qdrant_search_url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")
    return response.json()

@app.post("/upload-image/", summary="Analyze an image and save its description")
async def upload_image(
    secure: str = Depends(get_api_key),
    collection: str = Form(...),
    source: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = "Analyze this image in detail. If it contains text (like a diagram or knowledge map), transcribe it exactly. If it is a photo or illustration, describe its contents, style, and any notable features."
    
    gemini_vision_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": file.content_type, "data": image_b64}}
            ]
        }]
    }
    
    try:
        response = requests.post(gemini_vision_url, json=payload)
        response.raise_for_status()
        description_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini Vision API: {e}")

    metadata = MemoryMetadata(
        collection=collection,
        source=f"image_upload/{source}",
        timestamp=datetime.utcnow().isoformat() + "Z",
        tags=["image_upload", source]
    )
    memory_input = MemoryInput(text=description_text, metadata=metadata)
    
    return save_memory(memory_input, secure)
