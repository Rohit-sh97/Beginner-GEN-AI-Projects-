from sentence_transformers import SentenceTransformer
from app.console import print_status

print_status("⏳ [EMBEDDING] Loading sentence-transformer model")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print_status("✅ [EMBEDDING] Sentence-transformer model is ready")


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    print_status(f"🧠 [EMBEDDING] Creating vectors for {len(chunks)} chunk(s)")
    embeddings = model.encode(chunks,normalize_embeddings=True)
    vectors = embeddings.tolist()
    dimensions = len(vectors[0]) if vectors else 0
    print_status(
        f"✅ [EMBEDDING] Created {len(vectors)} vector(s) "
        f"with {dimensions} dimension(s)"
    )
    return vectors


def search_similar_texts(query: str, chunks: list[str], top_k: int = 2):
    import numpy as np

    print_status(
        f"🔎 [EMBEDDING SEARCH] Comparing query against "
        f"{len(chunks)} chunk(s); top_k={top_k}"
    )
    query_embedding = model.encode(query, normalize_embeddings=True
                                   )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    similarities = np.dot(chunk_embeddings, query_embedding)

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = [
        {
            "text": chunks[i],
            "similarity": round(float(similarities[i]), 4)
        }

        for i in ranked_indices
    ]
    print_status(f"✅ [EMBEDDING SEARCH] Returning {len(results)} result(s)")
    return results