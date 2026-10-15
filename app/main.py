import hashlib
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import settings
from app.models import AskRequest, AskResponse, DocumentInfo
from app.rag import RAGService

app = FastAPI(title="DocuMind AI API", version="1.0.0", description="Grounded answers from your PDFs")
service = RAGService(settings)


def get_service() -> RAGService:
    return service


@app.get("/health")
def health():
    return {"status": "ok", "ai_provider": settings.resolved_provider}


@app.post("/documents/upload", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), rag: RAGService = Depends(get_service)):
    if file.content_type != "application/pdf" or not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    content = await file.read(settings.max_file_size_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDFs must be {settings.max_file_size_mb} MB or smaller.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    document_id = hashlib.sha256(content).hexdigest()[:16]
    safe_name = Path(file.filename or "document.pdf").name
    path = settings.upload_path / f"{document_id}-{safe_name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    try:
        return rag.index_document(path, safe_name)
    except (ValueError, RuntimeError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents(rag: RAGService = Depends(get_service)):
    return rag.list_documents()


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, rag: RAGService = Depends(get_service)):
    if not rag.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: RAGService = Depends(get_service)):
    try:
        answer, sources = rag.answer(request.question)
        return AskResponse(answer=answer, sources=sources)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
