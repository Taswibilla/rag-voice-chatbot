import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import load_pdf, build_index, retrieve, generate_answer
from voice import transcribe_audio, text_to_speech

app = FastAPI(title="RAG Voice Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── In-memory index store ─────────────────────────────────
INDEX_STATE = {
    "index": None,
    "chunks": [],
    "metadata": [],
    "doc_names": []
}


# ── Route 1: Health check ─────────────────────────────────
@app.get("/")
def root():
    return {"status": "RAG Voice Chatbot API is running"}


# ── Route 2: Upload PDFs + Build Index ───────────────────
@app.post("/build-index")
async def build_index_route(
    files: list[UploadFile] = File(...),
    chunk_strategy: str = Form("overlap")
):
    docs = {}
    temp_dir = tempfile.mkdtemp()
    
    try:
        for uploaded_file in files:
            # Save uploaded PDF temporarily
            temp_path = os.path.join(temp_dir, uploaded_file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
            
            # Extract text
            text = load_pdf(temp_path)
            docs[uploaded_file.filename] = text
            print(f"Loaded {uploaded_file.filename}: {len(text)} characters")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Build FAISS index
    index, chunks, metadata = build_index(docs, strategy=chunk_strategy)
    
    # Save to global state
    INDEX_STATE["index"] = index
    INDEX_STATE["chunks"] = chunks
    INDEX_STATE["metadata"] = metadata
    INDEX_STATE["doc_names"] = list(docs.keys())
    
    return {
        "status": "success",
        "total_chunks": len(chunks),
        "documents": list(docs.keys()),
        "strategy_used": chunk_strategy
    }


# ── Route 3: Chat (text query) ────────────────────────────
class ChatRequest(BaseModel):
    query: str
    enable_tts: bool = True

@app.post("/chat")
def chat(req: ChatRequest):
    if INDEX_STATE["index"] is None:
        return {"error": "Index not built. Please upload PDFs first."}
    
    # Retrieve relevant chunks
    top_chunks = retrieve(req.query, INDEX_STATE["index"], INDEX_STATE["chunks"], k=4)
    
    # Generate answer
    answer = generate_answer(req.query, top_chunks)
    
    # Find source documents
    chunk_to_doc = dict(zip(INDEX_STATE["chunks"], INDEX_STATE["metadata"]))
    sources = list(set(chunk_to_doc.get(c, "unknown") for c in top_chunks))
    
    # TTS
    audio_b64 = None
    if req.enable_tts:
        audio_b64 = text_to_speech(answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "audio_b64": audio_b64
    }


# ── Route 4: Transcribe audio ─────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = transcribe_audio(audio_bytes)
    return {"text": text}


# ── Route 5: Index status ─────────────────────────────────
@app.get("/status")
def status():
    return {
        "index_built": INDEX_STATE["index"] is not None,
        "total_chunks": len(INDEX_STATE["chunks"]),
        "documents": INDEX_STATE["doc_names"]
    }