
from fastapi import FastAPI, File, HTTPException, UploadFile
from app.rag import retrieve_documents, retrieve_bm25_documents, ingest_document, rebuild_bm25_from_chroma
from app.rag import ingest_document, retrieve_documents
from app.hybrid_retriever import retrieve_hybrid
from app.chains import research_documents
from app.schemas import ResearchRequest, ResearchResponse
from app.evaluate import run_evaluation

app = FastAPI(
    title="AI Research Assistant",
    description="Research across your documents using RAG.",
    version="1.0.0",
)

@app.on_event("startup")
def initialize_retrieval():
    count = rebuild_bm25_from_chroma()
    print(f"BM25 initialized with {count} stored chunks.")

@app.get("/")
def home():
    return {
        "message": "Welcome to AI Research Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        chunk_count = ingest_document(filename, file_bytes)

        return {
            "message": "Document indexed successfully",
            "filename": filename,
            "chunks_created": chunk_count,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        print(f"UPLOAD ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check server logs.",
        ) from exc


@app.post("/documents/search")
def search_documents(question: str):
    try:
        results = retrieve_documents(question)

        return {
            "question": question,
            "results": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page"),
                }
                for doc in results
            ],
        }

    except Exception as exc:
        print(f"SEARCH ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Check server logs.",
        ) from exc

@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    try:
        return research_documents(request.question)

    except Exception as exc:
        print(f"RESEARCH ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Research generation failed. Check server logs.",
        ) from exc
        
        
        
@app.get("/debug/retrieval-comparison")
def retrieval_comparison(question: str):
    try:
        vector_results = retrieve_documents(question, k=4)
        bm25_results = retrieve_bm25_documents(question, k=4)
        hybrid_results = retrieve_hybrid(question, k=4)
 
        def format_results(documents):
            return [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "excerpt": doc.page_content[:300],
                }
                for doc in documents
            ]

        return {
            "question": question,
            "vector_search": format_results(vector_results),
            "bm25_search": format_results(bm25_results),
            "hybrid_search": format_results(hybrid_results),
        }

    except Exception as exc:
        print(
            f"RETRIEVAL COMPARISON ERROR: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Retrieval comparison failed. Check server logs.",
        ) from exc



@app.get("/debug/evaluate-retrieval")
def evaluate_retrieval():
    try:
        return run_evaluation()
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Retrieval evaluation failed. Check the server terminal.",
        )
