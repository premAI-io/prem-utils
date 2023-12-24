import together
from together.error import AttributeError, AuthenticationError, InstanceError, JSONError, RateLimitError, ResponseError

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class TogetherConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        together.api_key = api_key
        self.exception_mapping = {
            AuthenticationError: errors.PremProviderAuthenticationError,
            ResponseError: errors.PremProviderResponseValidationError,
            JSONError: errors.PremProviderResponseValidationError,
            InstanceError: errors.PremProviderAPIStatusError,
            RateLimitError: errors.PremProviderRateLimitError,
            AttributeError: errors.PremProviderResponseValidationError,
        }

    def parse_chunk(self, chunk):
        return {
            "id": None,
            "model": None,
            "object": None,
            "created": None,
            "choices": [
                {
                    "delta": {"content": str(chunk), "role": "assistant"},
                    "finish_reason": None,
                }
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
        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        try:
            if stream:
                response = together.Complete.create_streaming(
                    prompt=messages[-1]["content"],
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=presence_penalty,
                )
                return response
            else:
                response = together.Complete.create(
                    prompt=messages[-1]["content"],
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=presence_penalty,
                )
                plain_response = {
                    "id": str(response["id"]),
                    "choices": [
                        {
                            "finish_reason": None,
                            "index": None,
                            "message": {"content": choice["text"], "role": "assistant"},
                        }
                        for choice in response["output"]["choices"]
                    ],
                    "created": None,
                    "model": model,
                    "provider_name": "Together",
                    "provider_id": "together",
                }
                return plain_response
        except (
            AuthenticationError,
            ResponseError,
            JSONError,
            InstanceError,
            RateLimitError,
            AttributeError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="together", model=model, provider_message=str(error))
