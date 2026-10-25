from app.langchain_pipeline import (
    load_document,
    split_documents,
    store_documents,
    retrieve_documents,
)

file_path = input("Enter PDF or TXT path: ").strip().strip('"')

documents = load_document(file_path)
chunks = split_documents(documents)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

store_documents(chunks)
print("Chunks stored in LangChain ChromaDB!")

question = input("\nAsk a question about the document: ")
results = retrieve_documents(question)

print("\nRetrieved results:")
for result in results:
    print(f"\nSource: {result.metadata}")
    print(result.page_content[:500])
