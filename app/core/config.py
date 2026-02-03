import os
from dataclasses import dataclass
from typing import List, Optional


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _get_list(name: str) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_revision: Optional[str]
    trust_remote_code: bool
    device: str
    batch_size: int
    hf_token: Optional[str]
    cache_dir: Optional[str]
    image_size: Optional[int]
    max_image_bytes: int
    max_url_timeout: float
    allowlist_cidrs: List[str]
    denylist_cidrs: List[str]


DEFAULT_DENYLIST = [
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "169.254.169.254/32",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
]


def load_settings() -> Settings:
    return Settings(
        model_id=_get_env("MODEL_ID", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        model_revision=_get_env("MODEL_REVISION"),
        trust_remote_code=_get_bool("TRUST_REMOTE_CODE", False),
        device=_get_env("DEVICE", "cpu"),
        batch_size=_get_int("BATCH_SIZE", 1),
        hf_token=_get_env("HF_TOKEN"),
        cache_dir=_get_env("CACHE_DIR"),
        image_size=int(_get_env("IMAGE_SIZE", "0")) or None,
        max_image_bytes=_get_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024),
        max_url_timeout=_get_float("MAX_URL_TIMEOUT", 5.0),
        allowlist_cidrs=_get_list("ALLOWLIST_CIDRS"),
        denylist_cidrs=_get_list("DENYLIST_CIDRS") or DEFAULT_DENYLIST,
    )
