from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.console import print_status
from app.langchain_pipeline import (
    load_document,
    split_documents,
    store_documents,
    answer_question,
)

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    print_status(f"📥 [UPLOAD] Received file: {filename}")

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported",
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        # Save temporarily so the LangChain loader can read it.
        from pathlib import Path
        from tempfile import NamedTemporaryFile

        suffix = Path(filename).suffix.lower()

        with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = Path(temp.name)

        try:
            documents = load_document(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        # Keep the original filename in source metadata.
        for document in documents:
            document.metadata["source"] = filename

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the document",
            )

        chunks = split_documents(documents)
        store_documents(chunks)

        print_status(
            f"✅ [UPLOAD] {filename}: "
            f"{len(documents)} document(s), {len(chunks)} chunk(s)"
        )

        return {
            "filename": filename,
            "message": "Document uploaded and indexed successfully",
            "document_count": len(documents),
            "chunk_count": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print_status(f"⛔ [UPLOAD] Failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process document",
        ) from exc


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    print_status(f"❓ [ASK] {question}")

    try:
        result = answer_question(question)
        print_status("✅ [ASK] LangChain answer generated")
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as exc:
        print_status(f"⛔ [ASK] Failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question",
        ) from exc
