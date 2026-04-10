from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_youtube(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()
    return splitter.split_documents(docs)

def load_web(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return splitter.split_documents(docs)