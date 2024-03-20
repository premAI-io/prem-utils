import fireworks.client
from fireworks.client.error import (
    AuthenticationError,
    BadGatewayError,
    InternalServerError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ServiceUnavailableError,
)

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class FireworksAIConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        fireworks.client.api_key = api_key
        self.exception_mapping = {
            PermissionError: errors.PremProviderPermissionDeniedError,
            InvalidRequestError: errors.PremProviderUnprocessableEntityError,
            InternalServerError: errors.PremProviderInternalServerError,
            AuthenticationError: errors.PremProviderAuthenticationError,
            RateLimitError: errors.PremProviderRateLimitError,
            ServiceUnavailableError: errors.PremProviderAPIStatusError,
            BadGatewayError: errors.PremProviderAPIConnectionError,
        }

    def preprocess_request(self, messages):
        prompt = self.prompt_template
        system_prompt = ""
        user_prompt = messages[-1]["content"]
        for message in messages:
            if message["role"] == "system":
                system_prompt = f"{system_prompt} {message['content']}"
        prompt = prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)
        return prompt

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

    def preprocess_messages(self, messages):
        reorder_messages = []
        other_messages = []
        for message in messages:
            if message["role"] == "system":
                reorder_messages.append(message)
            else:
                other_messages.append(message)
        return reorder_messages + other_messages

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
        messages = self.preprocess_messages(messages)

        if max_tokens is None:
            max_tokens = 10000

        try:
            response = fireworks.client.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
            )
            if stream:
                return response
            else:
                plain_response = {
                    "choices": [
                        {
                            "finish_reason": choice.finish_reason,
                            "index": choice.index,
                            "message": {
                                "content": choice.message.content,
                                "role": choice.message.role,
                            },
                        }
                        for choice in response.choices
                    ],
                    "created": response.created,
                    "model": response.model,
                    "provider_name": "FireworksAI",
                    "provider_id": "fireworksai",
                    "usage": {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
                return plain_response
        except (
            RateLimitError,
            AuthenticationError,
            PermissionError,
            InvalidRequestError,
            InternalServerError,
            ServiceUnavailableError,
            BadGatewayError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="fireworksai", model=model, provider_message=str(error))
