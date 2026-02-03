<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc\img\logo_aisc_bmftr.jpg">
<h1> DINOv3 Embeddings API
</div>

Batch-first FastAPI service for generating image embeddings with DINOv3. Supports image URLs or base64 payloads, CPU/GPU execution, and Docker/Kubernetes deployment.

## Features

- **Batch-first embeddings**: Always send `images: [...]`, even for single-image requests
- **GPU-ready**: CUDA 13 base image and `DEVICE=cuda` toggle
- **Secure URL fetch**: SSRF protections with denylist/allowlist support
- **Ops-friendly**: Docker + K8s manifests and health checks

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, but recommended for faster performance)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd tool-dinov3-embeddings-api
   ```

2. Build and run with Docker:
   ```bash
   docker compose up --build
   ```

3. Access the application:
   - API: http://localhost:8000/v1/health

## User Guide

### Using the Tool
1. Check health:
   ```bash
   curl http://localhost:8000/v1/health
   ```
2. Request embeddings (single image = batch of 1):
   ```bash
   curl -X POST http://localhost:8000/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "images": [
         {"image_url": "https://example.com/image.jpg", "id": "img-1"}
       ]
     }'
   ```

### Recommendations
- For GPU, set `DEVICE=cuda` and run with the NVIDIA container runtime.
- Use `BATCH_SIZE` to cap request size and manage latency.


## Limitations

- **Model download time**: First start will download the model from Hugging Face.
- **URL inputs**: Only `http`/`https` are allowed, and internal IPs are blocked by default.


## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [DINOv3 model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## Author
- [AI Service Centre Berlin Brandenburg](https://hpi.de/kisz)

## License
See [LICENSE](LICENSE).

---

## Acknowledgements
<img src="00_aisc/img/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 01IS22092.
