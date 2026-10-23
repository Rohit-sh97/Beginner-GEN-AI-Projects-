from fastapi import FastAPI
from app.document_routes import router as document_router

app = FastAPI(title="AI Doc Chatbot")

app.include_router(document_router, prefix="/documents", tags=["Documents"])

@app.get("/")
def home():
    return {"message": "Welcome to the AI Document Chatbot!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}