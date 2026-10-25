
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(
    document_id: str,
    chunks: list[str],
    embeddings: list[list[float]]
):
    collection.add(
        ids=[f"{document_id}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"document_id": document_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
    )

    return collection.count()


def search_chunks(query_embedding: list[float], top_k: int = 3):
    count = collection.count()

    if count == 0:
        return {"documents": [], "distances": []}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count)
    )