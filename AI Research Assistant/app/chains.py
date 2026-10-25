import os 

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.prompts import research_prompt
from app.schemas import ResearchResponse

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

structured_llm = llm.with_structured_output(
    ResearchResponse,
    method="json_schema",
)

research_chain = research_prompt | structured_llm

def research_topic(topic: str):
    return research_chain.invoke({"topic": topic})