from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModel

from app.core.config import load_settings


@lru_cache(maxsize=1)
def load_model_and_processor() -> Tuple[AutoModel, AutoImageProcessor, torch.device]:
    settings = load_settings()
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
    )
    model.to(device)
    model.eval()
    return model, processor, device
