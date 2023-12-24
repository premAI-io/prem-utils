from collections.abc import Sequence

from anthropic import (
    AI_PROMPT,
    HUMAN_PROMPT,
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
        return {
            "id": chunk.log_id,
            "model": chunk.model,
            "object": None,
            "created": None,
            "choices": [
                {
                    "delta": {"content": chunk.completion, "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }

    def apply_prompt_template(self, messages):
        prompt = ""
        system_prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"{HUMAN_PROMPT} {message['content']}"
            elif message["role"] == "assistant":
                prompt += f"{AI_PROMPT} {message['content']}"
            if message["role"] == "system":
                system_prompt = f"{system_prompt} {message['content']}"
        return f"{system_prompt} \n {prompt} {AI_PROMPT} "

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
        prompt = self.apply_prompt_template(messages)
        if max_tokens is None:
            max_tokens = 10000
        try:
            response = self.client.completions.create(
                model=model,
                max_tokens_to_sample=max_tokens,
                prompt=prompt,
                stream=stream,
                top_p=top_p,
                temperature=temperature,
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
            "id": response.log_id,
            "choices": [
                {
                    "finish_reason": response.stop_reason,
                    "index": None,
                    "message": {"content": response.completion, "role": "assistant"},
                }
            ],
            "created": None,
            "model": response.model,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
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
