from django.conf import settings
from django.utils import timezone
from replicate import Client
from replicate.exceptions import ModelError, ReplicateError

from prem.gateway import exceptions
from prem.gateway.connectors.base import BaseConnector


class ReplicateConnector(BaseConnector):
    def __init__(self, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = Client(api_token=settings.REPLICATE_API_KEY)
        self.exception_mapping = {
            ReplicateError: exceptions.PremProviderInternalServerError,
            ModelError: exceptions.PremProviderAPIStatusError,
        }

    def parse_chunk(self, chunk):
        return {
            "id": None,
            "model": None,
            "object": None,
            "created": str(timezone.now()),
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
                response = self.client.stream(
                    ref=model,
                    input={"prompt": messages[-1]["content"]},
                )
                return response
            else:
                content = ""
                response = self.client.run(
                    ref=model,
                    input={"prompt": messages[-1]["content"]},
                )
                content = "".join([element for element in response])
        except (ReplicateError, ModelError) as error:
            custom_exception = self.exception_mapping.get(type(error), exceptions.PremProviderError)
            raise custom_exception(error, provider="replicate", model=model, provider_message=str(error))

        plain_response = {
            "id": None,
            "choices": [
                {
                    "finish_reason": None,
                    "index": None,
                    "message": {"content": content, "role": "assistant"},
                }
            ],
            "created": str(timezone.now()),
            "model": model,
            "provider_name": "Replicate",
            "provider_id": "replicate",
        }
        return plain_response
