from django.conf import settings
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from prem.gateway import exceptions
from prem.gateway.connectors.base import BaseConnector


class AzureOpenAIConnector(BaseConnector):
    def __init__(self, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_BASE_URL,
            api_version="2023-10-01-preview",
        )
        self.exception_mapping = {
            APIError: exceptions.PremProviderAPIErrror,
            PermissionDeniedError: exceptions.PremProviderPermissionDeniedError,
            UnprocessableEntityError: exceptions.PremProviderUnprocessableEntityError,
            InternalServerError: exceptions.PremProviderInternalServerError,
            AuthenticationError: exceptions.PremProviderAuthenticationError,
            BadRequestError: exceptions.PremProviderBadRequestError,
            NotFoundError: exceptions.PremProviderNotFoundError,
            RateLimitError: exceptions.PremProviderRateLimitError,
            APIResponseValidationError: exceptions.PremProviderAPIResponseValidationError,
            ConflictError: exceptions.PremProviderConflictError,
            APIStatusError: exceptions.PremProviderAPIStatusError,
            APITimeoutError: exceptions.PremProviderAPITimeoutError,
            APIConnectionError: exceptions.PremProviderAPIConnectionError,
        }

    def parse_chunk(self, chunk):
        return {
            "id": chunk.id,
            "model": chunk.model,
            "object": chunk.object,
            "created": chunk.created,
            "choices": [
                {
                    "delta": {"content": choice.delta.content, "role": choice.delta.role},
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
        tools: list[dict[str]] = None,
        tool_choice: dict = None,
    ):
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
            custom_exception = self.exception_mapping.get(type(error), exceptions.PremProviderError)
            raise custom_exception(error, provider="azure", model=model, provider_message=str(error))

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
            "provider_name": "Azure OpenAI",
            "provider_id": "azure_openai",
        }
        return plain_response
