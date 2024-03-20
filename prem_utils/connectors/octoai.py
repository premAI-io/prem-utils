from octoai.client import Client
from octoai.errors import OctoAIClientError, OctoAIServerError, OctoAIValidationError

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector

# NOTE deprecated connector


class OctoAIConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = Client(token=api_key)
        self.exception_mapping = {
            OctoAIClientError: errors.PremProviderPermissionDeniedError,
            OctoAIValidationError: errors.PremProviderUnprocessableEntityError,
            OctoAIServerError: errors.PremProviderInternalServerError,
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

        if "octoai" in model:
            model = model.replace("octoai/", "", 1)

        try:
            response = self.client.chat.completions.create(
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
        except (OctoAIClientError, OctoAIServerError, OctoAIValidationError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="octoai", model=model, provider_message=str(error))

        if stream:
            return response

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
            "provider_name": "OctoAI",
            "provider_id": "octoai",
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return plain_response
