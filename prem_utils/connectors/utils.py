from time import time
from uuid import uuid4

import tiktoken


def default_chatcompletion_response_id():
    return str(uuid4())


def default_chatcompletion_response_created():
    return int(round(time()))


def default_count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def aggregate_tokens_count(token_collection: list[dict | str] | str) -> int:
    tokens = 0
    if isinstance(token_collection, str):
        tokens = default_count_tokens(token_collection)
    elif isinstance(token_collection, list):
        if len(token_collection) > 0 and isinstance(token_collection[0], str):
            tokens = sum([default_count_tokens(h) for h in token_collection])
        elif len(token_collection) > 0 and isinstance(token_collection[0], dict):
            tokens = sum([default_count_tokens(h["content"]) for h in token_collection])
    return tokens


def default_chatcompletions_usage(history: list[dict | str] | str, completion: list[dict | str] | str):
    prompt_tokens = aggregate_tokens_count(history)
    completion_tokens = aggregate_tokens_count(completion)
    total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def default_embeddings_usage(text_to_embed: list[list[str]] | list[str] | str) -> dict:
    if isinstance(text_to_embed, list) and len(text_to_embed) > 0 and isinstance(text_to_embed[0], list):
        tokens = sum([aggregate_tokens_count(txt) for txt in text_to_embed])
    else:
        tokens = aggregate_tokens_count(text_to_embed)
    return {
        "prompt_tokens": tokens,
        "total_tokens": tokens,
    }
