# main.py (v2 - Now with image processing)
import os
import requests
import uuid
import base64
import shutil
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# ... (load_dotenv and Configuration sections remain the same) ...
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")

# ... (Security section remains the same) ...
api_key_header = APIKeyHeader(name="X-API-Key")
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# ... (Pydantic Models remain the same, we'll reuse them) ...
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

# ... (Helper functions remain the same) ...
def get_gemini_embedding(text: str):
    # ... (code for get_gemini_embedding is unchanged) ...
    try:
        gemini_embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GEMINI_API_KEY}"
        response = requests.post(gemini_embed_url, json={"model": "models/embedding-001", "content": {"parts": [{"text": text}]}})
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

# ... (Existing /save-memory/ and /search-memory/ endpoints remain the same) ...
@app.post("/save-memory/")
def save_memory(memory_input: MemoryInput, secure: str = Depends(get_api_key)):
    # ... (code for save_memory is unchanged) ...
    embedding = get_gemini_embedding(memory_input.text)
    try:
        collection_name = memory_input.metadata.collection
        qdrant_points_url = f"{QDRANT_URL}/collections/{collection_name}/points?wait=true"
        point_id = str(uuid.uuid4())
        payload = {"points": [{"id": point_id, "vector": embedding, "payload": memory_input.metadata.dict()}]}
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.put(qdrant_points_url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")
    return {"status": "success", "id": point_id, "collection": collection_name}

@app.post("/search-memory/")
def search_memory(search_input: SearchInput, secure: str = Depends(get_api_key)):
    # ... (code for search_memory is unchanged) ...
    query_embedding = get_gemini_embedding(search_input.query)
    try:
        collection_name = search_input.collection
        qdrant_search_url = f"{QDRANT_URL}/collections/{collection_name}/points/search"
        payload = {"vector": query_embedding, "limit": search_input.limit, "with_payload": True}
        headers = {"api-key": QDRANT_API_KEY}
        response = requests.post(qdrant_search_url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Qdrant API: {e}")
    return response.json()

# --- !! NEW IMAGE PROCESSING ENDPOINT !! ---
@app.post("/upload-image/", summary="Analyze an image and save its description to Qdrant")
async def upload_image(collection: str, source: str, file: UploadFile = File(...), secure: str = Depends(get_api_key)):
    # 1. Read image data
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # 2. Call Gemini Vision API to analyze the image
    prompt = "Analyze this image. If it contains text (like a diagram or knowledge map), extract it. If it is a piece of art or a photo, describe it in detail. Combine the extracted text and description into a comprehensive summary."
    
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

    # 3. Embed and save the description to Qdrant using our existing save_memory logic
    metadata = MemoryMetadata(
        collection=collection,
        source=f"image_upload/{source}",
        timestamp=datetime.utcnow().isoformat() + "Z",
        tags=["image_upload", source]
    )
    memory_input = MemoryInput(text=description_text, metadata=metadata)
    
    return save_memory(memory_input, secure)
