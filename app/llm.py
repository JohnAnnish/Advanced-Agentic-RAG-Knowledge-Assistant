from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import settings


llm = ChatOpenAI(
    model=settings.openai_chat_model,
    temperature=0,
    api_key=settings.openai_api_key,
)

embeddings = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    api_key=settings.openai_api_key,
)
