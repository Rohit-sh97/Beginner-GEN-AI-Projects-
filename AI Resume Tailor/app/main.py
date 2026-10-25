
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pypdf import PdfReader

from app.schemas import ResumeAnalysisRequest, ResumeAnalysisResponse
from app.chains import analyze_resume


app = FastAPI(
    title="AI Resume Tailor",
    description="Analyze a resume against a job description.",
    version="1.0.0",
)

MAX_PDF_SIZE = 5 * 1024 * 1024  # 5 MB


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Resume Tailor",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Existing endpoint: accepts resume text as JSON
@app.post("/analyze", response_model=ResumeAnalysisResponse)
def analyze(request: ResumeAnalysisRequest):
    try:
        return analyze_resume(
            request.resume_text,
            request.job_description,
        )
    except Exception as exc:
        print(f"ANALYSIS ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Resume analysis failed. Check server logs.",
        ) from exc


# New endpoint: accepts a PDF file
@app.post("/analyze-pdf", response_model=ResumeAnalysisResponse)
def analyze_pdf(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
):
    # 1. Check the uploaded filename
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    # 2. Validate the job description
    if len(job_description.strip()) < 30:
        raise HTTPException(
            status_code=422,
            detail="Job description must contain at least 30 characters.",
        )

    # 3. Read the uploaded PDF, with a size limit
    try:
        file_bytes = resume_file.file.read(MAX_PDF_SIZE + 1)

        if len(file_bytes) > MAX_PDF_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF must be 5 MB or smaller.",
            )

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"FILE READ ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file.",
        ) from exc

    # 4. Extract text from every PDF page
    try:
        pdf = PdfReader(BytesIO(file_bytes))

        if pdf.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported.",
            )

        resume_text = "\n".join(
            page.extract_text() or ""
            for page in pdf.pages
        ).strip()

    except HTTPException:
        raise
    except Exception as exc:
        print(f"PDF EXTRACTION ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=400,
            detail="Could not extract text. Please upload a valid PDF.",
        ) from exc

    # 5. Ensure the PDF contains enough extractable text
    if len(resume_text) < 50:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough readable text was found. "
                "Try a text-based PDF instead of a scanned image."
            ),
        )

    # 6. Reuse the same AI pipeline as /analyze
    try:
        return analyze_resume(
            resume_text,
            job_description,
        )
    except Exception as exc:
        print(f"PDF ANALYSIS ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Resume analysis failed. Check server logs.",
        ) from exc
