import argparse
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--requests", type=int, default=10)
    args = parser.parse_args()

    payload = {"images": [{"image_url": args.image_url, "id": "img-1"}]}
    latencies = []
    for _ in range(args.requests):
        start = time.perf_counter()
        response = requests.post(args.url, json=payload, timeout=30)
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000.0)

    avg = sum(latencies) / len(latencies)
    print(f"requests={len(latencies)} avg_ms={avg:.2f} p95_ms={sorted(latencies)[int(len(latencies)*0.95)-1]:.2f}")


if __name__ == "__main__":
    main()
