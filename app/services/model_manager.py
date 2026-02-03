from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModel

from app.core.config import load_settings


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@lru_cache(maxsize=1)
def load_model_and_processor() -> Tuple[AutoModel, AutoImageProcessor, torch.device, torch.dtype]:
    settings = load_settings()
    if settings.model_dtype:
        dtype = DTYPE_MAP.get(settings.model_dtype.lower())
        if dtype is None:
            raise ValueError(f"Unsupported MODEL_DTYPE: {settings.model_dtype}")
    else:
        dtype = torch.float16 if settings.device.startswith("cuda") else torch.float32

    device = torch.device(settings.device)

    processor = AutoImageProcessor.from_pretrained(
        settings.model_id,
        revision=settings.model_revision,
        trust_remote_code=settings.trust_remote_code,
        cache_dir=settings.cache_dir,
        token=settings.hf_token,
    )
    model = AutoModel.from_pretrained(
        settings.model_id,
        revision=settings.model_revision,
        trust_remote_code=settings.trust_remote_code,
        cache_dir=settings.cache_dir,
        token=settings.hf_token,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, processor, device, dtype
