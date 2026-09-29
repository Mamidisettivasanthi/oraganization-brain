import hashlib
import io
from pathlib import Path
import re
from typing import Any

from docx import Document as DocxDocument
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pypdf import PdfReader

from app.config import settings
from app.services.hindsight_client import HindsightClient
from app.services.hindsight_rag import HindsightRAG

router = APIRouter()
MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def _extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return "\n".join(
            page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages
        )
    if extension == ".docx":
        document = DocxDocument(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return content.decode("utf-8-sig")


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    doc_type: str | None = Form(None),
    tags: str | None = Form(None),
    source_reference: str | None = Form(None),
    document_id: str | None = Form(None),
):
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: pdf, docx, txt, md")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 15 MB)")
    if document_id and not re.fullmatch(r"[A-Za-z0-9._:-]{3,128}", document_id):
        raise HTTPException(422, "Document ID must be 3-128 letters, digits, '.', '_', ':', or '-'")
    if source_reference is not None and len(source_reference.strip()) > 300:
        raise HTTPException(422, "Source reference must be at most 300 characters")

    try:
        extracted = _extract_text(filename, content).strip()
    except Exception as exc:
        raise HTTPException(422, "Could not extract text from the uploaded file") from exc
    if not extracted:
        raise HTTPException(422, "No readable text was found in the uploaded file")

    metadata: dict[str, Any] = {
        "title": title or filename,
        "doc_type": doc_type or "GENERAL",
        "tags": [tag.strip() for tag in (tags or "").split(",") if tag.strip()],
        "document_id": document_id or hashlib.sha256(content).hexdigest(),
        "source_reference": source_reference.strip() if source_reference and source_reference.strip() else f"Uploaded file: {filename}",
        "version": "1",
    }
    try:
        result = await HindsightRAG().ingest_document(extracted, metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Document could not be retained in Hindsight"
        ) from exc
    return {
        "status": result.get("status", "stored"),
        "title": metadata["title"],
        "document_id": metadata["document_id"],
        "hindsight": result,
    }


@router.get("/documents")
async def list_documents():
    try:
        return await HindsightClient().list_documents(settings.HINDSIGHT_BANK_ID)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Could not list Hindsight documents"
        ) from exc