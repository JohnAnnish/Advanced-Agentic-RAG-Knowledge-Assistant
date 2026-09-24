from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS

from .agent import build_graph
from .cache import get_answer, save_answer
from .ingestion import load_documents
from .llm import embeddings
from .retrieval import HybridRetriever


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Advanced Agentic RAG Knowledge Assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for portfolio)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
graph = None


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    global retriever
    global graph

    documents = load_documents()

    if not documents:
        return {
            "status": "error",
            "message": "No PDFs found in data/documents/",
        }

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    retriever = HybridRetriever(vector_store)
    retriever.build_bm25(documents)
    graph = build_graph(retriever)

    return {
        "status": "ok",
        "documents": len(documents),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    if graph is None:
        return {
            "answer": "Please call /ingest first."
        }

    cached = get_answer(request.question)

    if cached:
        return {"answer": cached}

    result = graph.invoke(
        {
            "messages": [],
            "query": request.question,
            "documents": [],
            "grade": "",
            "attempts": 0,
        }
    )

    answer = result["messages"][-1].content

    save_answer(
        request.question,
        answer,
    )

    return {"answer": answer}
