from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.prompts import resume_prompt
from app.schemas import ResumeAnalysisResponse

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Use Groq's native JSON Schema output instead of tool calling.
structured_llm = llm.with_structured_output(
    ResumeAnalysisResponse,
    method="json_schema",
)

resume_chain = resume_prompt | structured_llm


def analyze_resume(resume_text: str, job_description: str):
    result = resume_chain.invoke(
        {
            "resume_text": resume_text,
            "job_description": job_description,
        }
    )

    if isinstance(result, dict):
        return ResumeAnalysisResponse(**result)

    return result