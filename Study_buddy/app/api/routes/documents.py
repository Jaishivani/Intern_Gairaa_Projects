"""Document upload and management endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pypdf.errors import PdfReadError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.guardrails import is_prompt_injection
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import Document, User
from app.schemas.document import DocumentResponse, DocumentTopicsResponse
from app.services.cache_service import clear_document_cache
from app.services.embedding_service import embed_texts
from app.services.pdf_service import extract_pdf_chunks
from app.services.topic_service import analyze_topics
from app.services.vector_store import add_document_chunks, delete_document_chunks, get_document_chunks

router = APIRouter()


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Document:
    """Upload a text-based PDF and index its page-aware chunks."""
    if not file.filename or file.content_type != "application/pdf" or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are supported.")
    settings = get_settings()
    pdf_bytes = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    if len(pdf_bytes) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF exceeds the upload size limit.")
    try:
        page_count, chunks = extract_pdf_chunks(pdf_bytes)
    except (ValueError, PdfReadError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    chunks = [chunk for chunk in chunks if not is_prompt_injection(chunk.text)]
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No safe, extractable text was found in the PDF.",
        )

    document = Document(filename=file.filename, page_count=page_count, owner_id=current_user.id)
    db.add(document)
    db.flush()
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    try:
        add_document_chunks(document.id, texts, [chunk.page for chunk in chunks], embeddings)
        db.commit()
    except Exception:
        db.rollback()
        delete_document_chunks(document.id)
        raise
    db.refresh(document)
    return document


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Document]:
    """List uploaded documents."""
    return list(
        db.scalars(
            select(Document)
            .where(Document.owner_id == current_user.id)
            .order_by(Document.created_at.desc())
        ).all()
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a document and its vector collection."""
    document = db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == current_user.id)
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    delete_document_chunks(document_id)
    db.delete(document)
    db.commit()
    clear_document_cache(document_id)


@router.get("/documents/{document_id}/topics", response_model=DocumentTopicsResponse)
def get_document_topics(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentTopicsResponse:
    """Return topic clusters and plotting coordinates for a document."""
    document = db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == current_user.id)
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    chunks = get_document_chunks(document_id)
    if not chunks:
        return DocumentTopicsResponse(topics=[], points=[])
    embeddings = embed_texts([str(chunk["text"]) for chunk in chunks])
    analysis = analyze_topics(chunks, embeddings, get_settings().topic_cluster_count)
    return DocumentTopicsResponse(**analysis)
