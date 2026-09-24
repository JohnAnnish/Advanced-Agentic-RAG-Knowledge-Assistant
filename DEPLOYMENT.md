# Deployment

GitHub does not make a repository live by itself.

For an always-live demo you need:
- an always-on Docker/API service
- managed PostgreSQL with pgvector
- managed Redis
- an LLM/embedding API key
- your document ingestion strategy

Expose `/health`, `/docs`, `/ingest`, and `/ask`.

The local starter uses FAISS for simplicity. For the final production deployment, replace the FAISS adapter with PostgreSQL + pgvector so embeddings persist across restarts.
