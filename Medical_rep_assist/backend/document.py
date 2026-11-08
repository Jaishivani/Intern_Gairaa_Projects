import re
from pathlib import Path

import fitz


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def clean_text(raw_text: str) -> str:
    cleaned = raw_text.replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned.strip()


def mask_common_pii(text: str) -> str:
    masked = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    masked = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", masked)
    masked = re.sub(r"\b(?:Patient ID|PatientId|MRN|MRN#?|ID)\s*[:#-]?\s*[A-Za-z0-9-]+\b", "[REDACTED_PATIENT_ID]", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b(?:Name|Patient Name)\s*[:\-]?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "[REDACTED_NAME]", masked)
    masked = re.sub(r"\b(?:dob|date of birth|birth date)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[REDACTED_DOB]", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b(?:Address|Street|City|Zip)\s*[:\-]?\s*.*?(?=\n|$)", "[REDACTED_ADDRESS]", masked, flags=re.IGNORECASE)
    return masked


def extract_text_from_pdf(file_path: str) -> str:
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_image(file_path: str) -> str:
    return "OCR placeholder: image text capture is not implemented in this simple MVP."


def process_report_document(file_path: str) -> str:
    if not file_path:
        raise ValueError("File path is required.")

    extension = Path(file_path).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type for document processing.")

    if extension == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    else:
        raw_text = extract_text_from_image(file_path)

    cleaned = clean_text(raw_text)
    masked = mask_common_pii(cleaned)
    return masked
