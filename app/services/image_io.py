import base64
import io
import socket
from ipaddress import ip_address, ip_network
from typing import Iterable
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

from app.core.config import load_settings


def _iter_resolved_ips(hostname: str) -> Iterable[str]:
    infos = socket.getaddrinfo(hostname, None)
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        yield sockaddr[0]


def _cidr_match(ip_str: str, cidrs: Iterable[str]) -> bool:
    ip = ip_address(ip_str)
    for cidr in cidrs:
        if ip in ip_network(cidr, strict=False):
            return True
    return False


def _validate_url(url: str) -> None:
    settings = load_settings()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    for ip_str in _iter_resolved_ips(parsed.hostname):
        if settings.allowlist_cidrs:
            if not _cidr_match(ip_str, settings.allowlist_cidrs):
                raise ValueError("URL host is not in allowlist")
        if _cidr_match(ip_str, settings.denylist_cidrs):
            raise ValueError("URL host is in denylist")


def _read_limited(stream: requests.Response, max_bytes: int) -> bytes:
    data = bytearray()
    for chunk in stream.iter_content(chunk_size=8192):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) > max_bytes:
            raise ValueError("Image exceeds MAX_IMAGE_BYTES")
    return bytes(data)


def load_image_from_url(url: str) -> Image.Image:
    settings = load_settings()
    _validate_url(url)
    response = requests.get(url, stream=True, timeout=settings.max_url_timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "image" not in content_type.lower():
        raise ValueError("URL did not return an image")
    data = _read_limited(response, settings.max_image_bytes)
    return _load_image_bytes(data)


def load_image_from_base64(payload: str) -> Image.Image:
    settings = load_settings()
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc
    if len(data) > settings.max_image_bytes:
        raise ValueError("Image exceeds MAX_IMAGE_BYTES")
    return _load_image_bytes(data)


def _load_image_bytes(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")
