
import pymupdf
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.text_splitter import split_text

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported"
        )

    content = await file.read()

    if filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        try:
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read this PDF file"
            )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the document"
        )

    chunks = split_text(text)

    return {
        "filename": filename,
        "text": text,
        "characters": len(text),
        "message": "Document read successfully",
        "chunk_count": len(chunks),
        "chunks": chunks
    }
