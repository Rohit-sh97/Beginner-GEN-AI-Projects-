
from pydantic import BaseModel, Field


class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(
        min_length=50,
        description="Text extracted from the resume",
    )
    job_description: str = Field(
        min_length=30,
        description="Target job description",
    )


class ResumeAnalysisResponse(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matching_skills: list[str]
    missing_skills: list[str]
    improvement_suggestions: list[str]
    tailored_summary: str
