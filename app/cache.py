import redis
from .config import settings


client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


def cache_key(question):
    return "rag:" + question.strip().lower()


def get_answer(question):
    return client.get(cache_key(question))


def save_answer(question, answer):
    client.setex(
        cache_key(question),
        3600,
        answer,
    )
