from collections.abc import Sequence

import cohere
from cohere.error import CohereAPIError, CohereConnectionError

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class CohereConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = cohere.Client(api_key)
        self.exception_mapping = {
            CohereAPIError: errors.PremProviderAPIErrror,
            CohereConnectionError: errors.PremProviderAPIConnectionError,
        }

    def preprocess_messages(self, messages):
        chat_history = []
        user_messages = []
        system_prompt = []
        for message in messages:
            if message["role"] == "user":
                user_messages.append(message["content"])
                chat_history.append({"user_name": "User", "text": message["content"]})
            elif message["role"] == "assistant":
                chat_history.append({"user_name": "Chatbot", "text": message["content"]})
            elif message["role"] == "system":
                system_prompt.append({"user_name": "System", "text": message["content"]})
        return system_prompt + chat_history, user_messages[-1]

    def parse_chunk(self, chunk):
        if hasattr(chunk, "text"):
            text = chunk.text
        else:
            text = ""
        return {
            "id": chunk.id,
            "model": None,
            "object": None,
            "created": None,
            "choices": [
                {
                    "delta": {"content": text, "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int = None,
        frequency_penalty: float = 0,
        log_probs: int = None,
        logit_bias: dict[str, float] = None,
        presence_penalty: float = 0,
        seed: int | None = None,
        stop: str | list[str] = None,
        stream: bool = False,
        temperature: float = 1,
        top_p: float = 1,
    ):
        chat_history, message = self.preprocess_messages(messages)
        try:
            response = self.client.chat(
                chat_history=chat_history,
                message=message,
                model="command",
                temperature=temperature,
                stream=stream,
            )
        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="cohere", model=model, provider_message=str(error))

        if stream:
            return response
        plain_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": response.text, "role": "assistant"},
                }
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": model,
            "provider_name": "Cohere",
            "provider_id": "cohere",
            "usage": {
                "completion_tokens": response.token_count["prompt_tokens"],
                "prompt_tokens": response.token_count["response_tokens"],
                "total_tokens": response.token_count["total_tokens"],
            },
        }
        return plain_response

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        try:
            texts = input if isinstance(input, list) else [input]
            response = self.client.embed(texts=texts, model=model, input_type="search_document")
        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="cohere", model=model, provider_message=str(error))
        return {
            "data": [{"index": index, "embedding": embedding} for index, embedding in enumerate(response.embeddings)],
            "model": model,
            "usage": connector_utils.default_embeddings_usage(texts),
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }
