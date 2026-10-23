import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your project root .env file."
    )

client = Groq(api_key=api_key)


def generate_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful document chatbot. "
                    "Answer using only the provided document context. "
                    "If the answer is not in the context, say you "
                    "could not find it in the documents. "
                    "Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""