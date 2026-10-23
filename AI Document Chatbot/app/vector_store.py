
import chromadb
from app.console import print_status

client = chromadb.PersistentClient(path="./chroma_db")
print_status("🗄️ [VECTOR STORE] Connected to persistent database at ./chroma_db")

collection = client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"}
)
print_status(
    f"✅ [VECTOR STORE] Collection 'document_chunks' ready "
    f"({collection.count()} existing chunk(s))"
)


def store_chunks(
    document_id: str,
    chunks: list[str],
    embeddings: list[list[float]]
):
    print_status(
        f"📤 [VECTOR STORE] Upserting {len(chunks)} chunk(s) "
        f"for document '{document_id}'"
    )
    collection.upsert(
        ids=[f"{document_id}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"document_id": document_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
    )

    total_chunks = collection.count()
    print_status(
        f"✅ [VECTOR STORE] Stored document '{document_id}'; "
        f"collection now has {total_chunks} chunk(s)"
    )
    return total_chunks

def search_chunks(query_embedding: list[float], top_k: int = 3):
    count = collection.count()
    print_status(
        f"🔎 [VECTOR STORE] Searching {count} stored chunk(s); "
        f"top_k={top_k}"
    )

    if count == 0:
        print_status("ℹ️ [VECTOR STORE] Search skipped: collection is empty")
        return {"documents": [], "distances": []}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count)
    )
    result_count = len(results["documents"][0]) if results["documents"] else 0
    print_status(f"✅ [VECTOR STORE] Search returned {result_count} matching chunk(s)")
    return results
