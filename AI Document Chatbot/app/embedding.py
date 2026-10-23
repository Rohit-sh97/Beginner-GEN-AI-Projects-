from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_embeddings(chunks: list[str]) -> list[list[float]]:
    embeddings = model.encode(chunks,normalize_embeddings=True)
    return embeddings.tolist()


def search_similar_texts(query: str, chunks: list[str], top_k: int = 2):
    import numpy as np

    query_embedding = model.encode(query, normalize_embeddings=True
                                   )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    similarities = np.dot(chunk_embeddings, query_embedding)

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "text": chunks[i],
            "similarity": round(float(similarities[i]), 4)
        }

        for i in ranked_indices
    ]