import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.connection import init_db, get_session
from src.rag.embeddings import embed_documents, similarity_search
from src.agent.react_agent import build_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="RAG Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []


@app.post("/documents/upload", summary="Upload and index a document")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename.endswith((".txt", ".pdf", ".md")):
        raise HTTPException(400, "Only .txt, .pdf, .md files are supported")

    content = await file.read()

    if file.filename.endswith(".pdf"):
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "Document is empty or could not be parsed")

    count = await embed_documents(
        session,
        [text],
        metadata={"filename": file.filename},
    )

    return {"message": f"Indexed {count} chunks from {file.filename}"}


@app.post("/chat", response_model=ChatResponse, summary="Chat with the RAG agent")
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    agent = build_agent(session)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent.invoke({"input": request.message})
    )

    sources = await similarity_search(session, request.message, k=3)

    return ChatResponse(
        answer=result["output"],
        sources=[{"content": s["content"][:200], "score": s["score"]} for s in sources],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
