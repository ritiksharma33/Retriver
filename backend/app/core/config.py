from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "media_vault"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()