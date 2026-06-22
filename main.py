from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from server import get_response
from utils.logger import get_logger
from authentication.api_key import validate_api_key

logger = get_logger(__name__)

app = FastAPI(
    title="LLM Server",
    description="Personal LLM server.",
    version="1.0.0",
)

class GenerateRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    model: str
    provider: str

class GenerateResponse(BaseModel):
    content: str
    model: str
    provider: str


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, x_api_key: str = Header(...)):
    entry = validate_api_key(x_api_key)

    if entry is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    logger.info("Incoming request — user: %s, project: %s, provider: %s, model: %s", entry["user"], entry["project"], request.provider, request.model)

    try:
        content = get_response(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            model=request.model,
            provider=request.provider,
        )
        logger.info("Request completed — user: %s, project: %s", entry["user"], entry["project"])
        return GenerateResponse(content=content, model=request.model, provider=request.provider)
    except ValueError as e:
        logger.error("Bad request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Internal error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))