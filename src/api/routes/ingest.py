from fastapi import APIRouter, UploadFile, File, Request, Query, HTTPException
from src.services.knowledge_base import KnowledgeBaseService
from src.core.sessions import SessionStore

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

@router.post("/ingest")
async def ingest_document(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID for scoping this knowledge base"),
    file: UploadFile = File(...),
):
    kb: KnowledgeBaseService = request.app.state.kb
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        chunks = kb.ingest_pdf(content, file.filename, chat_id=chat_id)
    else:
        text = content.decode("utf-8", errors="ignore")
        chunks = kb.ingest_text(text, chat_id=chat_id, metadata={"filename": file.filename, "type": "text"})
        
    return {"status": "success", "filename": file.filename, "chunks": chunks}

@router.post("/reset")
async def reset_kb(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID to clear"),
):
    kb: KnowledgeBaseService = request.app.state.kb
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")
    success = kb.clear_chat(chat_id)
    return {"status": "success" if success else "error"}

@router.get("/list")
async def list_documents(
    request: Request,
    chat_id: str = Query(..., description="Unique chat ID to list"),
):
    kb: KnowledgeBaseService = request.app.state.kb
    sessions: SessionStore = request.app.state.sessions
    if not await sessions.exists(chat_id):
        raise HTTPException(status_code=404, detail="Unknown or expired chat_id")
    documents = kb.list_all_documents(chat_id=chat_id)
    return {"status": "success", "documents": documents}
