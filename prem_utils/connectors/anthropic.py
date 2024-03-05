from collections.abc import Sequence

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector

# https://docs.anthropic.com/claude/reference/errors-and-rate-limits


class AnthropicConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = Anthropic(api_key=api_key)
        self.exception_mapping = {
            PermissionDeniedError: errors.PremProviderPermissionDeniedError,
            UnprocessableEntityError: errors.PremProviderUnprocessableEntityError,
            InternalServerError: errors.PremProviderInternalServerError,
            AuthenticationError: errors.PremProviderAuthenticationError,
            BadRequestError: errors.PremProviderBadRequestError,
            NotFoundError: errors.PremProviderNotFoundError,
            RateLimitError: errors.PremProviderRateLimitError,
            APIResponseValidationError: errors.PremProviderAPIResponseValidationError,
            ConflictError: errors.PremProviderConflictError,
            APIStatusError: errors.PremProviderAPIStatusError,
            APITimeoutError: errors.PremProviderAPITimeoutError,
            APIConnectionError: errors.PremProviderAPIConnectionError,
        }

    def parse_chunk(self, chunk):
        if hasattr(chunk, "delta") and hasattr(chunk, "index"):
            return {
                "id": chunk.index,
                "model": "anthropic-model",
                "object": None,
                "created": None,
                "choices": [
                    {
                        "delta": {"content": chunk.delta.text, "role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        else:
            return {
                "id": 0,
                "model": None,
                "object": None,
                "created": None,
                "choices": [
                    {
                        "delta": {"content": "", "role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }

    def preprocess_messages(self, messages):
        system_prompt = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                filtered_messages.append(message)
        return system_prompt, filtered_messages

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
        system_prompt, messages = self.preprocess_messages(messages)

        if max_tokens is None:
            max_tokens = 4096

        try:
            response = self.client.messages.create(
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                model=model,
                top_p=top_p,
                temperature=temperature,
                stream=stream,
            )
        except (
            NotFoundError,
            APIResponseValidationError,
            ConflictError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
            BadRequestError,
            APIConnectionError,
            AuthenticationError,
            InternalServerError,
            PermissionDeniedError,
            UnprocessableEntityError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="anthropic", model=model, provider_message=str(error))

        if stream:
            return response

        plain_response = {
            "choices": [
                {
                    "finish_reason": response.stop_reason,
                    "index": 0,
                    "message": {"content": response.content[0].text, "role": "assistant"},
                }
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": response.model,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
            "usage": connector_utils.default_chatcompletions_usage(messages, response.content[0].text),
        }
        return plain_response

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        return {
            "data": None,
            "model": None,
            "usage": None,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
        }
