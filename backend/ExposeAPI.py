from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from LLMInference import LLMInference
from IngestAudio import AudioIngestor
from IngestDocs import DocumentIngestor
from VectorStore import VectorStore
import numpy as np
import ollama

app = FastAPI()
audio_ingestor = AudioIngestor()
doc_ingestor = DocumentIngestor()
vector_store = VectorStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference = LLMInference()

class LoadModelRequest(BaseModel):
    model_name: str

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 256
    temperature: float = 0.7

class QueryResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "LLM Inference API is running"}

@app.get("/api/models")
async def get_models():
    models = inference.get_available_models()
    categorized = inference.get_categorized_models()
    return {"models": models, "categorized_models": categorized, "current_model": inference.llm_name}

@app.get("/api/trending-models")
async def get_popular_models():
    import requests
    import re
    
    models = {
        "popular": []
    }
    
    try:
        response_popular = requests.get("https://ollama.com/library?sort=popular", timeout=5)
        matches_popular = re.findall(r'href="/library/([^/"]+)"', response_popular.text)
        seen_popular = set()
        for match in matches_popular:
            if match not in seen_popular and match != "library":
                seen_popular.add(match)
                models["popular"].append(match)
                if len(models["popular"]) >= 8:
                    break
                    
        if len(models["popular"]) > 0:
            return models
            
    except Exception as e:
        print(f"Failed to fetch models from ollama.com: {e}")
        pass

    fallback = [
        "llama3.2", "mistral", "qwen2.5:0.5b", "gemma2", "phi3", "deepseek-coder-v2"
    ]
    return {"popular": fallback}

class DeleteModelRequest(BaseModel):
    model_name: str

@app.post("/api/models/delete")
async def delete_model(request: DeleteModelRequest):
    try:
        inference.delete_model(request.model_name)
        return {"status": "success", "message": f"Model {request.model_name} deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    from fastapi.responses import StreamingResponse
    import json
    import ollama
    
    available_models = inference.get_available_models()
    if request.model_name not in available_models:
        def pull_stream():
            try:
                for progress in ollama.pull(request.model_name, stream=True):
                    progress_dict = vars(progress) if hasattr(progress, '__dict__') else progress if isinstance(progress, dict) else dict(progress)
                    yield json.dumps(progress_dict) + "\n"
                
                inference.llm_name = request.model_name
                print(f"Model '{request.model_name}' ready.")
                yield json.dumps({"status": "success", "message": f"Model {request.model_name} loaded and ready."}) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                
        return StreamingResponse(pull_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})
    else:
        try:
            inference.load_model(request.model_name)
            return {"status": "success", "message": f"Model {request.model_name} loaded and ready."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

@app.post("/api/models/unload")
async def unload_model():
    try:
        inference.unload_model()
        return {"status": "success", "message": "Model unloaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    response_text = inference.generate(
        prompt=request.query,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"response": response_text}

@app.post("/api/audio/digest")
async def audio_digest(file: UploadFile = File(...), embedding_model: str = ...):
    import os
    from EmbedData import EmbeddingPipeline

    if not file.filename:
        return {"status": "error", "message": "No filename provided"}

    filename = file.filename
    file_location = f"temp_{filename}"

    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        transcribe_result = audio_ingestor.transcribe(file_location)
        text = str(transcribe_result["text"])

        prompt = f"Summarize this file in bullet points\n{text}"
        summary = inference.generate(prompt=prompt)

        # Embed transcription and store in vector store
        embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)
        chunks, embeddings = embedding_pipeline.process_document(text, filename)
        if len(chunks) > 0:
            vector_store.add_embeddings(embeddings, chunks)

        return {"transcribe": text, "summary": summary}
    finally:
        os.remove(file_location)


@app.get("/api/whisper/models")
async def get_whisper_models():
    available = audio_ingestor.get_available_models()
    downloaded = audio_ingestor.get_downloaded_models()
    return {"models": downloaded, "all_models": available, "current_model": audio_ingestor.model_name}


class WhisperModelRequest(BaseModel):
    model_name: str


@app.post("/api/whisper/models/load")
async def load_whisper_model(request: WhisperModelRequest):
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    import threading
    
    if request.model_name not in audio_ingestor.get_available_models():
        return {"status": "error", "message": f"Model {request.model_name} not available."}
    
    if audio_ingestor.model_name == request.model_name and audio_ingestor.model is not None:
        return {"status": "success", "message": f"Whisper model {request.model_name} already loaded."}

    model_sizes = {
        "tiny": 75, "base": 75, "small": 150, "medium": 300, "large": 400, "large-v2": 400, "large-v3": 400
    }
    total_steps = model_sizes.get(request.model_name, 200)

    def generate_progress():
        try:
            for step in range(1, total_steps + 1):
                progress = int((step / total_steps) * 100)
                yield json.dumps({"status": "downloading", "completed": progress, "total": 100, "message": f"Downloading Whisper model {request.model_name}..."}) + "\n"
                if step % 20 == 0:
                    import time
                    time.sleep(0.1)
            
            audio_ingestor.load_model(request.model_name)
            yield json.dumps({"status": "success", "completed": 100, "total": 100, "message": f"Whisper model {request.model_name} loaded successfully."}) + "\n"
        except Exception as e:
            import time
            time.sleep(0.1)
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(generate_progress(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/api/whisper/models/unload")
async def unload_whisper_model():
    try:
        audio_ingestor.unload_model()
        return {"status": "success", "message": "Whisper model unloaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class WhisperDeleteRequest(BaseModel):
    model_name: str


@app.post("/api/whisper/models/delete")
async def delete_whisper_model(request: WhisperDeleteRequest):
    try:
        import os
        import shutil
        
        if audio_ingestor.model_name == request.model_name:
            audio_ingestor.unload_model()
        
        cache_dir = os.path.expanduser("~/.cache/whisper")
        model_path = os.path.join(cache_dir, f"{request.model_name}.pt")
        
        if os.path.exists(model_path):
            os.remove(model_path)
            return {"status": "success", "message": f"Whisper model {request.model_name} deleted from cache."}
        else:
            return {"status": "success", "message": f"Whisper model {request.model_name} not found in cache (may already be deleted)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/documents/ingest")
async def ingest_document(file: UploadFile = File(...), embedding_model: str = ...):
    try:
        from EmbedData import EmbeddingPipeline
        import tempfile
        import os
        
        if not file.filename:
            return {"status": "error", "message": "No filename provided"}
        
        filename = file.filename
        embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            text = ""
            if filename.lower().endswith(('.txt', '.md')):
                with open(tmp_path, 'r') as f:
                    text = f.read()
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                text = doc_ingestor.perform_ocr(tmp_path)
            else:
                return {"status": "error", "message": f"Unsupported file type: {filename}"}
            
            if not text.strip():
                return {"status": "error", "message": "No text extracted from document"}
            
            # Process document: chunk and embed
            chunks, embeddings = embedding_pipeline.process_document(text, filename)
            
            if len(chunks) == 0:
                return {"status": "error", "message": "No chunks created from document"}
            
            # Add to vector store
            vector_store.add_embeddings(embeddings, chunks)
            
            return {
                "status": "success",
                "message": f"Document '{filename}' ingested successfully",
                "chunks_created": len(chunks),
                "doc_id": chunks[0]['doc_id'] if chunks else None
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        return {"status": "error", "message": str(e)}