
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from app.bm25_retriever import build_bm25_retriever, retrieve_bm25
from app.embeddings import embeddings


# Store the vector database inside the project folder.
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# Split documents into smaller, overlapping chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
)


# Create or reopen the persistent vector database.
vector_store = Chroma(
    collection_name="research_documents",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)



def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Keep only the highest-ranked chunk from each source page."""
    unique_documents = []
    seen_pages = set()

    for doc in documents:
        key = (
            doc.metadata.get("source", "Unknown"),
            doc.metadata.get("page"),
        )

        if key in seen_pages:
            continue

        seen_pages.add(key)
        unique_documents.append(doc)

    return unique_documents


def ingest_document(filename: str, file_bytes: bytes) -> int:
    """Extract, split, and store one PDF or TXT document."""

    documents = []

    if filename.lower().endswith(".pdf"):
        import io

        reader = PdfReader(io.BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                        },
                    )
                )

    elif filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="replace")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": 1,
                    },
                )
            )

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not documents:
        raise ValueError(
            "No readable text found. Scanned PDFs may need OCR."
        )

    chunks = text_splitter.split_documents(documents)

    vector_store.add_documents(chunks)

    stored_data = vector_store.get(include=["documents", "metadatas"])

    stored_chunks = [
        Document(
            page_content=content,
            metadata=metadata or {},
        )
        for content, metadata in zip(
            stored_data["documents"],
            stored_data["metadatas"],
        )
    ]

    build_bm25_retriever(stored_chunks)

    return len(chunks)


def retrieve_documents(question: str, k: int = 4):
    """Retrieve relevant chunks using vector search."""
    return vector_store.similarity_search(
        query=question,
        k=k,
    )
    
def retrieve_documents(question: str, k: int = 4) -> list[Document]:
    # Fetch extra candidates so deduplication can still return k results.
    candidates = vector_store.similarity_search(
        query=question,
        k=k * 3,
    )

    unique_documents = deduplicate_documents(candidates)
    return unique_documents[:k]



def retrieve_bm25_documents(
    question: str,
    k: int = 4,
) -> list[Document]:
    # Retrieve extra candidates before removing duplicates.
    candidates = retrieve_bm25(question, k=k * 3)

    unique_documents = deduplicate_documents(candidates)

    return unique_documents[:k]

def rebuild_bm25_from_chroma():
    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    stored_chunks = [
        Document(
            page_content=content,
            metadata=metadata or {},
        )
        for content, metadata in zip(
            stored_data["documents"],
            stored_data["metadatas"],
        )
        if content
    ]

    build_bm25_retriever(stored_chunks)
    return len(stored_chunks)

