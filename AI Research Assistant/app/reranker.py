from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(
    question: str,
    documents: list[Document],
    top_n: int = 4,
) -> list[Document]:
    """Rank retriveved documrnts by relevance to the quetion"""
    
    if not documents:
        return []
    
    pairs = [
        (question, documents.page_content)
        for documents in documents
    ]
    
    scores = reranker.predict(pairs)
    
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    
    return [
        document
        for document, score in ranked_documents[:top_n]
    ]