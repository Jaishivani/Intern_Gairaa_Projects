import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from auth import get_current_user, router as auth_router
from database import Report, create_tables, get_db
from document import process_report_document

create_tables()

app = FastAPI(title="Medical Report AI Assistant", version="0.1.0")
app.include_router(auth_router)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Medical Report AI Assistant backend is running",
    }


@app.post("/reports/upload", status_code=status.HTTP_201_CREATED)
async def upload_report(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.filename is None or file.filename == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file is required.",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: PDF, PNG, JPG, JPEG.",
        )

    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES.get(extension, set()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content type does not match the uploaded extension.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large. Maximum size is 10 MB.",
        )

    upload_directory = Path(__file__).resolve().parent / "uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)

    safe_file_name = f"{uuid.uuid4().hex}{extension}"
    saved_path = upload_directory / safe_file_name

    with saved_path.open("wb") as destination:
        destination.write(file_bytes)

    report = Report(
        user_id=current_user.id,
        file_name=file.filename,
        file_path=str(saved_path),
        upload_date=datetime.utcnow(),
        status="uploaded",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "message": "Report uploaded successfully.",
        "report_id": report.id,
        "file_name": file.filename,
        "stored_file_name": safe_file_name,
        "status": report.status,
    }


@app.post("/reports/{report_id}/process")
def process_report(
    report_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id).first()
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    text = process_report_document(report.file_path)
    return {
        "report_id": report.id,
        "message": "Personal information has been protected before AI processing.",
        "processed_text": text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
