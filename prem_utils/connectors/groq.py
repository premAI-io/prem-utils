import logging

from groq import Groq
from groq._exceptions import (
    APIConnectionError,
    APIError,
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
from prem_utils.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class GroqConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.api_key = api_key
        self.exception_mapping = {
            APIConnectionError: errors.PremProviderAPIConnectionError,
            APIError: errors.PremProviderAPIErrror,
            APIStatusError: errors.PremProviderAPIStatusError,
            APITimeoutError: errors.PremProviderAPITimeoutError,
            PermissionDeniedError: errors.PremProviderPermissionDeniedError,
            NotFoundError: errors.PremProviderNotFoundError,
            UnprocessableEntityError: errors.PremProviderUnprocessableEntityError,
            InternalServerError: errors.PremProviderInternalServerError,
            AuthenticationError: errors.PremProviderAuthenticationError,
            RateLimitError: errors.PremProviderRateLimitError,
            BadRequestError: errors.PremProviderAPIConnectionError,
            ConflictError: errors.PremProviderConflictError,
            APIResponseValidationError: errors.PremProviderAPIResponseValidationError,
        }
        self.client = Groq(api_key=api_key)

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
                    "finish_reason": choice.finish_reason,
                }
                for choice in chunk.choices
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
        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        if "groq" in model:
            model = model.replace("groq/", "", 1)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
            )
        except (
            APIConnectionError,
            APIError,
            APIStatusError,
            APITimeoutError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
            InternalServerError,
            AuthenticationError,
            RateLimitError,
            BadRequestError,
            ConflictError,
            APIResponseValidationError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))

        if stream:
            return response
        plain_response = {
            "id": response.id,
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "message": {"content": choice.message.content, "role": choice.message.role},
                }
                for choice in response.choices
            ],
            "created": response.created,
            "model": response.model,
            "provider_name": "OpenAI",
            "provider_id": "openai",
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return plain_response
