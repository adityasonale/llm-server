"""
OpenAI-compatible HTTP surface: POST /v1/chat/completions.

Any OpenAI client works against it — the OpenAI SDK, LangChain's
ChatOpenAI(base_url=...), curl. Auth accepts the OpenAI convention
(`Authorization: Bearer <key>`) as well as this server's `X-API-Key`.
Errors use OpenAI's shape, {"error": {"message", "type", "code"}}, so SDKs
parse them into their usual exception types.
"""
from fastapi import APIRouter, Depends, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from authentication.api_key import validate_api_key
from chat import chat_completion, chat_completion_stream
from providers.base import ProviderOverloadedError
from providers.openai_compatible import UpstreamError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])


class OpenAIError(Exception):
    def __init__(self, status: int, message: str, type_: str, code: str | None = None, headers: dict | None = None):
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.headers = headers or {}


def _error_response(status: int, message: str, type_: str, code: str | None = None, headers: dict | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
        headers=headers,
    )


def register_error_handlers(app) -> None:
    @app.exception_handler(OpenAIError)
    async def _openai_error(_: Request, e: OpenAIError):
        return _error_response(e.status, e.message, e.type, e.code, e.headers)

    # FastAPI's default 422 {"detail": [...]} is kept for /generate; /v1 gets
    # OpenAI's 400 shape so SDK clients report it as a BadRequestError.
    from fastapi.exception_handlers import request_validation_exception_handler

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, e: RequestValidationError):
        if request.url.path.startswith("/v1/"):
            first = e.errors()[0] if e.errors() else {}
            location = ".".join(str(p) for p in first.get("loc", []) if p != "body")
            return _error_response(400, f"Invalid request: {location}: {first.get('msg', 'invalid')}", "invalid_request_error")
        return await request_validation_exception_handler(request, e)


async def require_api_key(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> dict:
    key = x_api_key
    if key is None and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()

    entry = validate_api_key(key) if key else None
    if entry is None:
        raise OpenAIError(401, "Invalid or missing API key.", "authentication_error", "invalid_api_key")
    return entry


class ChatCompletionRequest(BaseModel):
    # Only what the gateway itself reads is declared; everything else (tools,
    # tool_choice, temperature, response_format, stream_options, ...) is kept
    # as-is and forwarded to the provider.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict]
    stream: bool = False


@router.get("/models")
async def list_models(entry: dict = Depends(require_api_key)):
    """Models registered in models.yaml, with the format and backend each would use on this machine."""
    from model_manager import MANAGER
    try:
        return {"object": "list", "data": MANAGER.describe()}
    except ValueError as e:
        raise OpenAIError(500, str(e), "server_error")


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, entry: dict = Depends(require_api_key)):
    body = request.model_dump()
    logger.info(
        "Incoming chat completion — user: %s, project: %s, model: %s, stream: %s, tools: %s",
        entry["user"], entry["project"], request.model, request.stream, bool(body.get("tools")),
    )

    try:
        if request.stream:
            stream = await chat_completion_stream(body)
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await chat_completion(
            body,
            langsmith_extra={"metadata": {"user": entry["user"], "project": entry["project"]}},
        )

    except UpstreamError as e:
        if e.payload:
            # Already OpenAI-shaped by the upstream — forward it verbatim.
            return JSONResponse(status_code=e.status, content=e.payload, headers=e.headers)
        raise OpenAIError(e.status, e.message, "upstream_error", headers=e.headers)
    except ProviderOverloadedError as e:
        logger.warning("Provider overloaded: %s", e)
        raise OpenAIError(503, str(e), "server_overloaded", headers={"Retry-After": str(e.retry_after)})
    except ValueError as e:
        logger.error("Bad request: %s", e)
        raise OpenAIError(400, str(e), "invalid_request_error")
    except Exception as e:
        logger.error("Internal error: %s", e, exc_info=True)
        raise OpenAIError(500, "Internal server error.", "server_error")
