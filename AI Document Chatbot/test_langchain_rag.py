from app.langchain_pipeline import answer_question

question = input("Ask a question about your stored document: ")

result = answer_question(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(source)
