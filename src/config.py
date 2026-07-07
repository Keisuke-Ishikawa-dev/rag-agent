from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    database_url: str = "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
