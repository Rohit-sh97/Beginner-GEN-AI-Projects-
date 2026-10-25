from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    topic: str = Field(
        min_length=3,
        description="Topic to research"
    )
    
class ResearchResponse(BaseModel):
        topic: str
        overiview: str 
        key_points: list[str]
        conclusion: str