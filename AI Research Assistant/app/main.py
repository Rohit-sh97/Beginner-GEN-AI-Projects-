from fastapi import FastAPI, HTTPException

from app.schemas import ResearchRequest, ResearchResponse
from app.chains import research_topic

app = FastAPI(
    title="AI Research Assistant",
    description="Generate structured research overviews with AI.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Research Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    try:
        return research_topic(request.topic)
    except Exception as exc:
        print(f"RESEARCH ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Research generation failed. Check server logs.",
        ) from exc