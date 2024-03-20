from collections.abc import Sequence

from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException, MistralConnectionException
from mistralai.models.chat_completion import ChatMessage

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class MistralConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = MistralClient(api_key=api_key)
        self.exception_mapping = {
            MistralAPIException: errors.PremProviderAPIStatusError,
            MistralConnectionException: errors.PremProviderAPIConnectionError,
        }

    def parse_chunk(self, chunk):
        return {
            "id": chunk.id,
            "model": chunk.model,
            "object": chunk.object,
            "created": chunk.created,
            "choices": [
                {
                    "delta": {
                        "content": choice.delta.content,
                        "role": choice.delta.role,
                    },
                    "finish_reason": None,
                }
                for choice in chunk.choices
            ],
        }

    def build_messages(self, messages):
        chat_messages = []
        for message in messages:
            if message["role"] != "system":
                chat_message = ChatMessage(content=message["content"], role=message["role"])
                chat_messages.append(chat_message)
        return chat_messages

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
        messages = self.build_messages(messages)
        try:
            if stream:
                response = self.client.chat_stream(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response
            else:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                plain_response = {
                    "choices": [
                        {
                            "finish_reason": str(choice.finish_reason),
                            "index": choice.index,
                            "message": {
                                "content": choice.message.content,
                                "role": choice.message.role,
                            },
                        }
                        for choice in response.choices
                    ],
                    "created": connector_utils.default_chatcompletion_response_created(),
                    "model": response.model,
                    "provider_name": "Mistral",
                    "provider_id": "mistralai",
                    "usage": {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
                return plain_response
        except (MistralAPIException, MistralConnectionException) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="mistralai", model=model, provider_message=str(error))

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        try:
            response = self.client.embeddings(
                model=model,
                input=input if type(input) is list else [input],
            )
            return {
                "data": [{"index": emb.index, "embedding": emb.embedding} for emb in response.data],
                "model": response.model,
                "usage": {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "provider_name": "Mistral",
                "provider_id": "mistralai",
            }
        except (MistralAPIException, MistralConnectionException) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="mistralai", model=model, provider_message=str(error))
