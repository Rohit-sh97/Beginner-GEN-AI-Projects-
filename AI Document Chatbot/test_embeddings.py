
from app.embedding import search_similar_texts

chunks = [
    "I build web applications using React and Next.js.",
    "I develop frontend websites using JavaScript.",
    "The weather is sunny today."
]

query = "Which frontend technologies do I know?"

results = search_similar_texts(query, chunks, top_k=2)

for result in results:
    print("Text:", result["text"])
    print("Similarity:", result["similarity"])
    print("-" * 40)
