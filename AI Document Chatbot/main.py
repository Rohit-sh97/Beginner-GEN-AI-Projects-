from fastapi import FastAPI
from app.console import print_status
from app.document_routes import router as document_router

app = FastAPI(title="AI Doc Chatbot")

app.include_router(document_router, prefix="/documents", tags=["Documents"])


@app.get("/")
def home():
    print_status("🏠 [API] Home route requested: GET /")
    return {"message": "Welcome to the AI Document Chatbot!"}


@app.get("/health")
def health_check():
    print_status("💚 [API] Health check requested: GET /health")
    return {"status": "healthy"}