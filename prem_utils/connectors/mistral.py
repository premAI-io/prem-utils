import json
from collections.abc import Sequence

from mistralai.async_client import MistralAsyncClient
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
        self.async_client = MistralAsyncClient(api_key=api_key)
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

    def _get_arguments(self, arguments):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return None

    async def chat_completion(
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
        tools=None,
    ):
        if tools is not None and stream:
            raise errors.PremProviderError(
                "Cannot use tools with stream=True",
                provider="mistralai",
                model=model,
                provider_message="Cannot use tools with stream=True",
            )
        messages = self.build_messages(messages)

        request_data = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens if max_tokens != 0 else None,
            temperature=temperature,
            top_p=top_p,
            random_seed=seed,
            tools=tools,
        )
        try:
            if stream:
                # Client actually returns an AsyncIterator,
                # not a coroutine, so there's no need to await it
                return self.async_client.chat_stream(**request_data)

            response = self.client.chat(**request_data)

            plain_response = {
                "choices": [
                    {
                        "finish_reason": choice.finish_reason.value if choice.finish_reason else None,
                        "index": choice.index,
                        "message": {
                            "content": choice.message.content,
                            "role": choice.message.role,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "arguments": self._get_arguments(tool_call.function.arguments),
                                        "name": tool_call.function.name,
                                    },
                                    "type": tool_call.type.value if tool_call.type else None,
                                }
                                for tool_call in choice.message.tool_calls
                            ]
                            if choice.message.tool_calls
                            else None,
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


class MistralAzureConnector(MistralConnector):
    def __init__(self, api_key: str, endpoint: str, prompt_template: str = None):
        super().__init__(api_key=api_key, prompt_template=prompt_template)
        self.client = MistralClient(endpoint=endpoint, api_key=api_key)
        self.async_client = MistralAsyncClient(endpoint=endpoint, api_key=api_key)
        self.exception_mapping = {
            MistralAPIException: errors.PremProviderAPIStatusError,
            MistralConnectionException: errors.PremProviderAPIConnectionError,
        }
