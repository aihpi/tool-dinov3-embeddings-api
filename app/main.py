import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.core.logging import setup_logging
from app.services.embeddings import warmup


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("DISABLE_WARMUP") or os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("warmup_skipped")
        yield
        return
    try:
        warmup()
    except Exception as exc:
        logger.error("warmup_failed", extra={"error": str(exc)})
    yield


app = FastAPI(title="DINOv3 Embeddings API", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
