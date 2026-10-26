
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

bm25_retriever = None


def build_bm25_retriever(documents: list[Document]):
    global bm25_retriever

    # Remove duplicate chunks using source, page, and content.
    unique_documents = []
    seen = set()

    for doc in documents:
        key = (
            doc.metadata.get("source", "Unknown"),
            doc.metadata.get("page"),
            doc.page_content,
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(doc)

    if not unique_documents:
        bm25_retriever = None
        return

    bm25_retriever = BM25Retriever.from_documents(unique_documents)
    bm25_retriever.k = 4


def retrieve_bm25(question: str, k: int = 4) -> list[Document]:
    if bm25_retriever is None:
        return []

    # Fetch extra candidates so duplicates can be removed.
    bm25_retriever.k = k
    return bm25_retriever.invoke(question)
