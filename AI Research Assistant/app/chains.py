import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.prompts import rag_prompt
from app.schemas import ResearchResponse
from app.rag import retrieve_documents

# Load variables from the project-root .env file.
load_dotenv()

# Check that the API key exists without displaying it.
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your project-root .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

def research_documents(question: str) -> ResearchResponse:
    # 1. Retrieve relevant chunks from Chroma.
    documents = retrieve_documents(question, k=4)

    if not documents:
        return ResearchResponse(
            question=question,
            answer="No relevant information was found in the uploaded documents.",
            sources=[],
        )

    # 2. Prepare the retrieved text for the LLM.
    context_parts = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source {index}: {source}, Page {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # 3. Generate an answer grounded in retrieved context.
    response = (rag_prompt | llm).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # 4. Return the answer with source references.
    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "excerpt": doc.page_content[:300],
        }
        for doc in documents
    ]

    return ResearchResponse(
        question=question,
        answer=response.content,
        sources=sources,
    )