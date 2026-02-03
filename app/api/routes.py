import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import EmbeddingsRequest, EmbeddingsResponse, HealthResponse
from app.core.config import load_settings
from app.services.embeddings import compute_embeddings
from app.services.image_io import load_image_from_base64, load_image_from_url
from app.services.model_manager import load_model_and_processor

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = load_settings()
    try:
        load_model_and_processor()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(
        status="ok",
        model_id=settings.model_id,
        device=settings.device,
        model_loaded=model_loaded,
        version="0.1.0",
    )


@router.post("/embeddings", response_model=EmbeddingsResponse)
def embeddings(request: EmbeddingsRequest, http_request: Request) -> EmbeddingsResponse:
    settings = load_settings()
    if len(request.images) > settings.batch_size:
        raise HTTPException(status_code=400, detail="Batch size exceeds BATCH_SIZE")

    images: List[object] = []
    ids: List[str] = []
    for item in request.images:
        try:
            if item.image_url:
                image = load_image_from_url(item.image_url)
            else:
                image = load_image_from_base64(item.image_base64 or "")
            images.append(image)
            ids.append(item.id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Failed to fetch image") from exc

    start = time.perf_counter()
    embeddings = compute_embeddings(images)
    latency_ms = (time.perf_counter() - start) * 1000.0

    logger.info(
        "embeddings_computed",
        extra={
            "request_id": getattr(http_request.state, "request_id", None),
            "batch_size": len(images),
            "latency_ms": round(latency_ms, 2),
        },
    )

    dim = len(embeddings[0]) if embeddings else 0
    return EmbeddingsResponse(
        embeddings=embeddings,
        dim=dim,
        model_id=load_settings().model_id,
        ids=ids if any(ids) else None,
        latency_ms=latency_ms,
    )
