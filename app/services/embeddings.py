import logging
from typing import List

import torch

from app.core.config import load_settings
from app.services.model_manager import load_model_and_processor

logger = logging.getLogger(__name__)


def _pool_output(outputs: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state.mean(dim=1)
    raise ValueError("Model output does not contain pooler_output or last_hidden_state")


def compute_embeddings(images: List[object]) -> List[List[float]]:
    settings = load_settings()
    model, processor, device = load_model_and_processor()
    processor_kwargs = {"return_tensors": "pt"}
    if settings.image_size:
        processor_kwargs["size"] = {"shortest_edge": settings.image_size}

    inputs = processor(images=images, **processor_kwargs)
    inputs = {k: v.to(device=device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pooled = _pool_output(outputs)
        if not torch.isfinite(pooled).all():
            logger.warning("non_finite_embeddings_detected")
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    embeddings = pooled.detach().cpu().float().tolist()
    return embeddings


def warmup() -> None:
    model, processor, device = load_model_and_processor()
    model_dtype = next(model.parameters()).dtype
    dummy = torch.zeros((1, 3, 224, 224), device=device, dtype=model_dtype)
    with torch.no_grad():
        outputs = model(pixel_values=dummy)
        pooled = _pool_output(outputs)
    logger.info(
        "warmup_complete",
        extra={
            "tensor_shape": list(pooled.shape),
            "tensor_dtype": str(pooled.dtype),
        },
    )
