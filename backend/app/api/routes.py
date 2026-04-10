from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.loader import load_youtube, load_web
from app.services.embedder import ingest_documents
from app.services.chain import get_chain, memory

router = APIRouter()

class IngestRequest(BaseModel):
    url: str
    source_type: str  # "youtube" | "web"

class ChatRequest(BaseModel):
    question: str

@router.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        docs = load_youtube(req.url) if req.source_type == "youtube" else load_web(req.url)
        count = ingest_documents(docs)
        return {"status": "ok", "chunks_added": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        chain = get_chain()
        result = chain({"question": req.question})
        sources = [
            {"content": d.page_content[:200], "source": d.metadata.get("source", "")}
            for d in result.get("source_documents", [])
        ]
        return {"answer": result["answer"], "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_memory():
    memory.clear()
    return {"status": "memory cleared"}