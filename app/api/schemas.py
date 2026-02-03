from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ImageItem(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    id: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self) -> "ImageItem":
        sources = [self.image_url, self.image_base64]
        if sum(1 for source in sources if source) != 1:
            raise ValueError("Exactly one of image_url or image_base64 is required")
        return self


class EmbeddingsRequest(BaseModel):
    images: List[ImageItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_images(self) -> "EmbeddingsRequest":
        if not self.images:
            raise ValueError("images must contain at least one item")
        return self


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    model_id: str
    ids: Optional[List[Optional[str]]] = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    model_loaded: bool
    version: str
