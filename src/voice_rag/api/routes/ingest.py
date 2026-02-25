from fastapi import APIRouter, UploadFile, File, Request
from src.voice_rag.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

@router.post("/ingest")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    kb: KnowledgeBaseService = request.app.state.kb
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        chunks = kb.ingest_pdf(content, file.filename)
    else:
        text = content.decode("utf-8", errors="ignore")
        chunks = kb.ingest_text(text, {"filename": file.filename})
        
    return {"status": "success", "filename": file.filename, "chunks": chunks}

@router.post("/reset")
async def reset_kb(request: Request):
    kb: KnowledgeBaseService = request.app.state.kb
    success = kb.clear_all()
    return {"status": "success" if success else "error"}
