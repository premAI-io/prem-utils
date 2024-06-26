from collections.abc import Sequence
from uuid import uuid4

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
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
        self.async_client = AsyncAnthropic(api_key=api_key)
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

    def _get_content(self, response, tools=False):
        if not tools:
            return {"content": response.content[0].text, "role": "assistant", "tool_calls": []}
        tool_messages = filter(lambda x: hasattr(x, "input") and hasattr(x, "name"), response.content)
        return {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "id": str(uuid4()),
                    "function": {
                        "arguments": tool_message.input,
                        "name": tool_message.name,
                    },
                    "type": "function",
                }
                for tool_message in tool_messages
            ],
        }

    def _parse_tools(self, tools: list[dict[str, any]]):
        if not tools:
            return []
        parsed_tools = []

        for tool in tools:
            transformed_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", None),
                "input_schema": tool["function"]["parameters"],
            }
            parsed_tools.append(transformed_tool)

        return parsed_tools

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
        tools = self._parse_tools(tools)
        if tools != [] and stream:
            raise errors.PremProviderError(
                "Cannot use tools with stream=True",
                provider="anthropic",
                model=model,
                provider_message="Cannot use tools with stream=True",
            )
        system_prompt, messages = self.preprocess_messages(messages)

        if max_tokens is None or max_tokens == 0:
            max_tokens = 4096

        request_data = dict(
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            model=model,
            top_p=top_p,
            temperature=temperature,
            stream=stream,
            stop_sequences=stop,
            tools=tools,
        )
        try:
            if stream:
                return await self.async_client.messages.create(**request_data)

            response = self.client.messages.create(**request_data)
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
        plain_response = {
            "choices": [
                {
                    "finish_reason": response.stop_reason,
                    "index": 0,
                    "message": self._get_content(response, tools=len(tools) > 0),
                }
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": response.model,
            "provider_name": "Anthropic",
            "provider_id": "anthropic",
            "usage": connector_utils.default_chatcompletions_usage(
                messages, response.content[0].text if len(tools) < 1 else ""
            ),
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
