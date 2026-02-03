# Integrations

## LiteLLM
Use this service as a custom embedding provider by pointing LiteLLM to `/v1/embeddings` and mapping the input to a batch list.

- Set the provider base URL to your service host.
- Ensure LiteLLM sends batch-first requests.

## Qdrant
Store embeddings in Qdrant with the returned `embeddings` array and `ids` list. For batch calls, map each embedding to its corresponding payload metadata.
