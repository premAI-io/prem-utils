import json
import logging

from groq import AsyncGroq, Groq
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
        self.async_client = AsyncGroq(api_key=api_key)

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
                provider="groq",
                model=model,
                provider_message="Cannot use tools with stream=True",
            )
        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        if "groq" in model:
            model = model.replace("groq/", "", 1)

        request_data = dict(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens if max_tokens != 0 else None,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            logprobs=log_probs,
            logit_bias=logit_bias,
            top_p=top_p,
            tools=tools,
        )

        try:
            if stream:
                return await self.async_client.chat.completions.create(**request_data)

            response = self.client.chat.completions.create(**request_data)
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

        plain_response = {
            "id": response.id,
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
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
                                "type": tool_call.type,
                            }
                            for tool_call in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None,
                    },
                }
                for choice in response.choices
            ],
            "created": response.created,
            "model": response.model,
            "provider_name": "Groq",
            "provider_id": "groq",
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return plain_response
