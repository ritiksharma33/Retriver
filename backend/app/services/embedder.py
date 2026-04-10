import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GEMINI_API_KEY
)

def get_vectorstore():
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PATH
    )

def ingest_documents(docs):
    vs = get_vectorstore()
    vs.add_documents(docs)
    vs.persist()
    return len(docs)