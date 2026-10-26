
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(
        min_length=3,
        description="Question to answer using uploaded documents",
    )


class SourceReference(BaseModel):
    source: str
    page: int | None = None
    excerpt: str


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
