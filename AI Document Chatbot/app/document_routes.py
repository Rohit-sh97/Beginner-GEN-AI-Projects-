from app.vector_store import store_chunks, search_chunks
from app.embedding import create_embeddings
from app.console import print_status
import pymupdf
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.text_splitter import split_text
from pydantic import BaseModel
from app.llm import generate_answer


router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    print_status(f"📥 [UPLOAD] Received file: {filename or '(no filename)'}")

    if not filename.lower().endswith((".txt", ".pdf")):
        print_status(f"⛔ [UPLOAD] Rejected unsupported file: {filename}")
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported"
        )

    content = await file.read()
    print_status(f"📦 [UPLOAD] Read {len(content)} bytes from {filename}")

    if filename.lower().endswith(".txt"):
        print_status("📄 [READ] Decoding TXT file as UTF-8")
        text = content.decode("utf-8", errors="replace")
    else:
        try:
            print_status("📕 [READ] Extracting text from PDF pages")
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                page_count = len(pdf)
                text = "\n".join(page.get_text() for page in pdf)
            print_status(f"📖 [READ] Extracted text from {page_count} PDF page(s)")
        except Exception:
            print_status(f"⛔ [READ] Could not read PDF: {filename}")
            raise HTTPException(
                status_code=400,
                detail="Could not read this PDF file"
            )

    if not text.strip():
        print_status(f"⛔ [READ] No readable text found in {filename}")
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the document"
        )

    print_status(f"✅ [READ] Extracted {len(text)} characters from {filename}")
    print_status("✂️ [SPLIT] Sending extracted text to the text splitter")
    chunks = split_text(text)
    print_status(f"✅ [SPLIT] Created {len(chunks)} text chunk(s)")

    document_id = filename

    print_status("🧠 [EMBED] Sending chunks to the embedding model")
    embeddings = create_embeddings(chunks)
    print_status(f"✅ [EMBED] Received {len(embeddings)} embedding(s)")

    print_status(f"🗄️ [STORE] Saving chunks and embeddings for {document_id}")
    total_chunks = store_chunks(document_id, chunks, embeddings)
    print_status(
        f"✅ [UPLOAD] Pipeline complete for {filename}; "
        f"collection has {total_chunks} chunk(s)"
    )

    return {
        "filename": filename,
        "text": text,
        "characters": len(text),
        "message": "Document read successfully",
        "chunk_count": len(chunks),
        "chunks": chunks
    }


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()
    print_status(f"❓ [ASK] Received question: {question}")

    if not question:
       raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    print_status("🧠 [EMBED] Converting question to embedding")
    query_embedding = create_embeddings([question])[0]
    print_status("✅ [EMBED] Question embedding created")

    print_status("🔎 [SEARCH] Searching for relevant chunks")
    results = search_chunks(query_embedding, top_k=3)

    if not results["documents"] or not results["documents"][0]:
        print_status("ℹ️ [SEARCH] No relevant chunks found")
        return {
            "question": question,
            "answer": "I'm sorry, I could not find the answer in the provided document context.",
            "sources": []
        }

    matches = []

    for text, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        matches.append({
            "text": text,
            "distance": round(distance, 4),
            "document_id": metadata["document_id"]
        })

    context = "\n\n".join(
        match["text"] for match in matches
    )

    print_status("🤖 [LLM] Generating answer with Groq")
    answer = generate_answer(question, context)
    print_status("✅ [LLM] Answer generated")


    return {
             "question": question,
             "answer": answer,
             "sources": [
             {
                      "document_id": match["document_id"],
                      "text": match["text"]
             }
             for match in matches
        ]
    }
    