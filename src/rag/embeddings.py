import asyncio
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.config import settings

embeddings_model = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


async def embed_documents(session: AsyncSession, texts: List[str], metadata: dict = {}) -> int:
    chunks = splitter.create_documents(texts, metadatas=[metadata] * len(texts))

    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(
        None,
        lambda: embeddings_model.embed_documents([c.page_content for c in chunks])
    )

    count = 0
    for chunk, vector in zip(chunks, vectors):
        await session.execute(
            text("""
                INSERT INTO documents (content, metadata, embedding)
                VALUES (:content, :metadata, :embedding)
            """),
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": str(vector),
            }
        )
        count += 1

    await session.commit()
    return count


async def similarity_search(session: AsyncSession, query: str, k: int = None) -> List[dict]:
    k = k or settings.retriever_k

    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(
        None,
        lambda: embeddings_model.embed_query(query)
    )

    result = await session.execute(
        text("""
            SELECT content, metadata, 1 - (embedding <=> :query_vec::vector) AS score
            FROM documents
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :k
        """),
        {"query_vec": str(query_vector), "k": k}
    )

    return [
        {"content": row.content, "metadata": row.metadata, "score": float(row.score)}
        for row in result.fetchall()
    ]
