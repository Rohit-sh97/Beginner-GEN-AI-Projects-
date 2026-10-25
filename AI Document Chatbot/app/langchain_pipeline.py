import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            return [
                Document(
                    page_content=page.get_text(),
                    metadata={"source": path.name, "page": page.number + 1},
                )
                for page in pdf
                if page.get_text().strip()
            ]
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": path.name})]

    raise ValueError(f"Unsupported file type: {suffix}")


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="langchain_document_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def store_documents(chunks: list[Document]) -> None:
    if not chunks:
        raise ValueError("No chunks to store.")

    ids = [
        f"{chunk.metadata['source']}_{chunk.metadata.get('page', 0)}_{i}"
        for i, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(documents=chunks, ids=ids)


def retrieve_documents(question: str, top_k: int = 3) -> list[Document]:
    return vector_store.similarity_search(question, k=top_k)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful document chatbot. "
            "Answer using only the provided document context. "
            "If the answer is not in the context, say you could not "
            "find it in the documents. Do not invent facts.\n\n"
            "Document context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

answer_chain = prompt | llm | StrOutputParser()


def answer_question(question: str) -> dict:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": "I could not find the answer in the provided documents.",
            "sources": [],
        }

    context = "\n\n".join(document.page_content for document in documents)
    answer = answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": answer,
        "sources": [
            {
                "text": document.page_content,
                **document.metadata,
            }
            for document in documents
        ],
    }
