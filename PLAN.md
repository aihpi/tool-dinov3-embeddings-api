# Implementation Plan: DINOv3 Embeddings API

## 0. Goals & Scope
- Provide a production-ready embeddings service for DINOv3 vision models.
- Expose a simple HTTP API to embed images from URL or base64.
- Support CPU/GPU execution, batching, and predictable latency.
- Include containerization, Kubernetes manifests, and minimal docs.

## 1. Repository Structure
- Create a clean layout:
  - `app/` (FastAPI service)
  - `app/core/` (config, logging, model loader)
  - `app/api/` (routes, schemas)
  - `app/services/` (image fetch/preprocess, inference)
  - `tests/` (unit + integration)
  - `docker/` (Dockerfile(s))
  - `k8s/` (deployment/service/namespace)
  - `scripts/` (dev utilities)
  - `docs/` (API spec, usage, integration notes)

## 2. API Contract
- Versioning: prefix all endpoints with `/v1` (e.g., `/v1/embeddings`).
- Batch-first input: always send a list of images. Single-image use cases send a batch of size 1.

### Endpoint: `POST /v1/embeddings`
- Inputs (one of `image_url` or `image_base64` per item):
  - `images`: list of objects:
    - `image_url` (string, optional)
    - `image_base64` (string, optional)
    - `id` (string, optional client correlation id)
- Validation:
  - Each item must include exactly one of `image_url` or `image_base64`.
  - Enforce max payload size and URL allowlist/denylist.
- Response:
  - `embeddings`: list[list[float]]
  - `dim` (int)
  - `model_id` (string)
  - `ids` (list of echoed ids, optional)
  - `latency_ms` (float)

### Endpoint: `GET /v1/health`
- Response: status, model loaded, device, version

## 3. Configuration
- Environment variables:
  - `MODEL_ID` (default model)
  - `MODEL_REVISION` (pin model revision or commit)
  - `TRUST_REMOTE_CODE` (HF models that require it)
  - `MODEL_DTYPE` (fp16/bf16/fp32 override)
  - `DEVICE` (cpu/cuda)
  - `BATCH_SIZE` (default 1)
  - `HF_TOKEN` (optional)
  - `CACHE_DIR` (model cache)
  - `IMAGE_SIZE` (default resize)
  - `MAX_IMAGE_BYTES`, `MAX_URL_TIMEOUT`
- Config module reads env, validates, and logs resolved values.

## 4. Model Loading & Inference
- Load DINOv3 with `transformers` + `torch`.
- Pin model dtype based on device (fp16 for GPU, fp32 for CPU) unless overridden by `MODEL_DTYPE`.
- Warmup pass on startup to validate model.
- Implement simple request batching (single-process, queue + timeout).

## 5. Image Fetch & Preprocessing
- Support URL fetch with timeouts, size limits, content-type check.
- Support base64 decode with size limit.
- Normalize with model-specific mean/std pulled from model config when available.
- Resize and center-crop to model input size from config when available.
- Handle EXIF orientation where possible.
- Log final tensor shape and dtype during warmup.

## 6. Service Skeleton
- FastAPI app with routers and Pydantic schemas.
- Error handling:
  - 400 for validation errors
  - 422 for decode/transform errors
  - 502 for upstream fetch errors
- Structured logging (JSON) with request id.
- SSRF protection for `image_url`:
  - Denylist internal/private IPs (e.g., 169.254.169.254, RFC1918 ranges).
  - Optional allowlist for demo environments.

## 7. Containerization
- Dockerfile:
  - Base image: python + torch runtime
  - Install deps (`fastapi`, `uvicorn`, `transformers`, `pillow`, `requests`)
  - Set `HF_HOME`/`TRANSFORMERS_CACHE`
  - Expose port 8000
- Provide `docker-compose.yml` for local run (optional GPU profile).

## 8. Kubernetes Manifests
- Deployment with:
  - resource requests/limits
  - optional GPU `nodeSelector`/`tolerations`
  - env vars for model + device
  - liveness/readiness probes
- Optional PVC for model cache to reduce re-downloads on restarts.
- Service (ClusterIP) + optional Ingress stub.

## 9. Observability
- Add request timing and model inference timing.
- Health endpoint includes model/device info.

## 10. Testing
- Unit tests:
  - config parsing
  - image decode/validation
  - preprocess output shape
- Integration tests:
  - `/health` responds
  - `/embeddings` with base64
  - `/embeddings` with URL (mocked)

## 11. Documentation
- `README.md`: quickstart, env vars, example curl
- `docs/api.md`: API contract + examples
- `docs/integrations.md`: LiteLLM (custom embedding provider) + Qdrant flow notes

## 12. Validation & Handoff
- Provide a minimal load test script (e.g., `scripts/load_test.py`).
- Add sample images for tests (small, permissive license).
- Tag a release version once API is stable.
