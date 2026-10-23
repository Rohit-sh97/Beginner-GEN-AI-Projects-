
from app.embedding import create_embeddings
from app.vector_store import store_chunks, search_chunks

chunks = [
    "I build web applications using React and Next.js.",
    "I develop frontend websites using JavaScript.",
    "The weather is sunny today."
]

# Convert chunks into vectors
embeddings = create_embeddings(chunks)

# Store the chunks and vectors in ChromaDB
count = store_chunks("test_doc", chunks, embeddings)
print("Total stored chunks:", count)

# Convert the question into a vector
query = "Which frontend technologies do I know?"
query_embedding = create_embeddings([query])[0]

# Search ChromaDB
results = search_chunks(query_embedding, top_k=2)

for text, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print("\nRelevant text:", text)
    print("Distance:", round(distance, 4))
