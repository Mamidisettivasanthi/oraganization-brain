from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Feedback
from app.schemas import FeedbackRequest, FeedbackResponse
from app.services.ingestion import ingest_feedback_document
from app.utils import parse_uuid

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """👍/👎 + expert correction. Corrections are stored AND indexed as retrievable
    knowledge, so future answers use them (with citations)."""
    fb = Feedback(
        conversation_id=parse_uuid(request.conversation_id),
        message_id=parse_uuid(request.message_id),
        question=request.question,
        original_answer=request.original_answer,
        rating=request.rating,
        expert_correction=request.correction,
        expert_name=request.expert_name,
        status="approved",  # human-submitted → validated
    )
    db.add(fb)
    db.flush()

    doc_id = None
    if request.correction and request.rating == "incorrect":
        doc_id = ingest_feedback_document(
            db, request.question, request.original_answer,
            request.correction, request.expert_name or "expert")
        fb.document_id = parse_uuid(doc_id)

    db.commit()
    return FeedbackResponse(
        feedback_id=str(fb.id), status="approved",
        correction_indexed=doc_id is not None,
        message=("Correction stored and indexed — future answers will use it."
                 if doc_id else "Feedback recorded. Thank you!"))