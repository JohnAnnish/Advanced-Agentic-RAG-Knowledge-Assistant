# Advanced Agentic RAG Knowledge Assistant

Portfolio-ready Agentic RAG with FastAPI, LangGraph, hybrid BM25 + vector retrieval, RRF, MMR-ready retrieval, reranking, query rewriting, retrieval grading, citations, Redis caching, PostgreSQL/pgvector architecture, RAGAS starter, and Docker.

## Local demo
1. Copy `.env.example` to `.env` and add your API key.
2. Put PDFs in `data/documents/`.
3. Run `docker compose up --build`.
4. Open `http://localhost:8000/docs`.
5. Call `POST /ingest`.
6. Call `POST /ask`.

## Live demo
GitHub stores code but does not run it. For an always-live portfolio demo, deploy the Docker container to an always-on cloud service and use managed PostgreSQL/pgvector and Redis. Configure the environment variables from `.env.example`.

The starter uses FAISS for an easy local demo. The retrieval adapter is isolated so it can be replaced by persistent PostgreSQL + pgvector for the production deployment.
