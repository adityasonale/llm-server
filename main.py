from contextlib import asynccontextmanager

from fastapi import FastAPI

from chat import close_upstreams
from openai_api import router as openai_router, register_error_handlers


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_upstreams()


app = FastAPI(
    title="LLM Server",
    description="Personal LLM server.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(openai_router)
register_error_handlers(app)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
