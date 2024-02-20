from collections.abc import Sequence

import numpy as np
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
from anthropic.types import Message

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
        if max_tokens is None:
            max_tokens = 10000

        messages_np = np.asarray(messages)
        system_idx = np.asarray([x["role"] == "system" for x in messages])
        system = messages_np[system_idx][0]["content"] if any(system_idx) else ""
        non_system_messages = messages_np[~system_idx]
        try:
            response: Message = self.client.messages.create(
                model=model,
                messages=non_system_messages,
                max_tokens=max_tokens,
                stream=stream,
                top_p=top_p,
                temperature=temperature,
                system=system,
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
                    "index": index,
                    "message": {"content": content.text, "role": "assistant"},
                }
                for index, content in enumerate(response.content)
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": response.model,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
        return {
            "data": None,
            "model": None,
            "usage": None,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
        }
