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
