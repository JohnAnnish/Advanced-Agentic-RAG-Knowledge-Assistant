from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing_extensions import TypedDict

from .config import settings
from .llm import llm
from .retrieval import HybridRetriever


class AgentState(TypedDict):
    messages: Annotated[list, lambda old, new: old + new]
    query: str
    documents: list
    grade: str
    attempts: int


class Grade(BaseModel):
    relevant: Literal["yes", "no"]


grader = llm.with_structured_output(Grade)

rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Rewrite this question into a precise search query.
    Keep the original meaning. Do not answer it.

    Question:
    {question}
    """
)

answer_prompt = ChatPromptTemplate.from_template(
    """
    You are an enterprise knowledge assistant.

    Answer using only the supplied context.
    If the context is insufficient, say so.

    Cite sources in this format:
    [filename, page N]

    Question:
    {question}

    Context:
    {context}
    """
)


def build_graph(retriever: HybridRetriever):

    def retrieve(state: AgentState):
        documents = retriever.retrieve(state["query"])
        documents = retriever.rerank(
            state["query"],
            documents,
        )

        return {
            "documents": documents,
            "attempts": state["attempts"] + 1,
        }

    def grade(state: AgentState):
        context = "\n\n".join(
            document.page_content
            for document in state["documents"]
        )

        result = grader.invoke(
            f"""
            Question:
            {state["query"]}

            Retrieved context:
            {context}

            Are these documents sufficient and relevant?
            """
        )

        return {"grade": result.relevant}

    def rewrite(state: AgentState):
        response = (
            rewrite_prompt | llm
        ).invoke(
            {"question": state["query"]}
        )

        return {"query": response.content}

    def generate(state: AgentState):
        context = []

        for document in state["documents"]:
            source = document.metadata.get(
                "source", "unknown"
            )
            page = document.metadata.get(
                "page", "unknown"
            )

            context.append(
                f"Source: {source}, Page: {page}\n"
                f"{document.page_content}"
            )

        response = (
            answer_prompt | llm
        ).invoke(
            {
                "question": state["query"],
                "context": "\n\n".join(context),
            }
        )

        return {
            "messages": [
                AIMessage(content=response.content)
            ]
        }

    def route(state: AgentState):
        if state["grade"] == "yes":
            return "generate"

        if state["attempts"] >= settings.max_attempts:
            return "generate"

        return "rewrite"

    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    builder.add_conditional_edges(
        "grade",
        route,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
