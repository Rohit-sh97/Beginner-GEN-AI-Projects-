
from app.evaluation_dataset import EVALUATION_DATASET
from app.rag import retrieve_documents, retrieve_bm25_documents
from app.hybrid_retriever import retrieve_hybrid


def is_relevant(document, expected_sources):
    source = document.metadata.get("source", "Unknown")
    page = document.metadata.get("page")

    return (source, page) in expected_sources


def evaluate_retriever(retriever, top_k=4):
    total = len(EVALUATION_DATASET)
    hits = 0
    results = []

    for test_case in EVALUATION_DATASET:
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]

        documents = retriever(question, k=top_k)

        hit = any(
            is_relevant(document, expected_sources)
            for document in documents
        )

        if hit:
            hits += 1

        results.append({
            "question": question,
            "hit": hit,
            "expected_sources": [
                {"source": source, "page": page}
                for source, page in expected_sources
            ],
            "retrieved_sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page"),
                }
                for doc in documents
            ],
        })

    return {
        "hit_rate_at_4": round(hits / total, 2) if total else 0,
        "hits": hits,
        "total_questions": total,
        "details": results,
    }



def evaluate_mrr(retriever, top_k=4):
    reciprocal_ranks = []
    details = []

    for test_case in EVALUATION_DATASET:
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]

        documents = retriever(question, k=top_k)

        reciprocal_rank = 0.0
        correct_rank = None

        for rank, document in enumerate(documents, start=1):
            if is_relevant(document, expected_sources):
                reciprocal_rank = 1 / rank
                correct_rank = rank
                break

        reciprocal_ranks.append(reciprocal_rank)

        details.append({
            "question": question,
            "first_correct_rank": correct_rank,
            "reciprocal_rank": round(reciprocal_rank, 3),
        })

    return {
        "mrr_at_4": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 3
        ) if reciprocal_ranks else 0,
        "details": details,
    }


def run_evaluation():
    retrievers = {
        "vector_search": retrieve_documents,
        "bm25_search": retrieve_bm25_documents,
        "hybrid_search": retrieve_hybrid,
    }

    results = {}

    for name, retriever in retrievers.items():
        results[name] = {
            **evaluate_retriever(retriever),
            **evaluate_mrr(retriever),
        }

    return results
