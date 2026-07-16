import os

from pypdf import PdfReader

MAX_CHARS = 4000


def extract_text(file_path: str, filename: str) -> str:
    """Best-effort text extraction from an uploaded resume. Returns '' on any failure
    rather than raising, since a resume is optional context, not a required input."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return ""

    return text.strip()[:MAX_CHARS]
