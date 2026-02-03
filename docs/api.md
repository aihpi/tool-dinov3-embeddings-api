# API Contract

Base URL: `http://<host>:8000`

## POST /v1/embeddings
Batch-first embeddings endpoint.

### Request
```json
{
  "images": [
    {"image_url": "https://example.com/image.jpg", "id": "img-1"},
    {"image_base64": "<base64>", "id": "img-2"}
  ]
}
```

### Response
```json
{
  "embeddings": [[0.1, 0.2], [0.3, 0.4]],
  "dim": 1024,
  "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
  "ids": ["img-1", "img-2"],
  "latency_ms": 12.3
}
```

## GET /v1/health
```json
{
  "status": "ok",
  "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
  "device": "cpu",
  "model_loaded": true,
  "version": "0.1.0"
}
```
