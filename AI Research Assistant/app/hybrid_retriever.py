
from langchain_core.documents import Document
from app.reranker import rerank_documents
from app.rag import retrieve_documents, retrieve_bm25_documents, deduplicate_documents


def reciprocal_rank_fusion(
    vector_results: list[Document],
    bm25_results: list[Document],
    k: int = 60,
    top_n: int = 4,
) -> list[Document]:
    """Combine vector and BM25 rankings using RRF."""

    scores = {}
    documents = {}

    ranked_lists = [vector_results, bm25_results]

    for results in ranked_lists:
        for rank, doc in enumerate(results, start=1):
            # Identify a chunk by its content and source metadata.
            key = (
                doc.metadata.get("source", "Unknown"),
                doc.metadata.get("page"),
                doc.page_content,
            )

            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            documents[key] = doc

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [documents[key] for key in ranked_keys[:top_n]]




def retrieve_hybrid(question: str, k: int = 4) -> list[Document]:
    """Retrieve, fuse, deduplicate, and rerank relevant pages."""

    vector_results = retrieve_documents(question, k=10)
    bm25_results = retrieve_bm25_documents(question, k=10)

    candidates = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_n=20,
    )

    # Keep one candidate per source page before reranking.
    candidates = deduplicate_documents(candidates)

    return rerank_documents(
        question=question,
        documents=candidates,
        top_n=k,
    )
