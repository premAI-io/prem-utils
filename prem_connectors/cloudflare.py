import json
import logging

import requests
from django.conf import settings
from django.utils import timezone

from prem.gateway import exceptions
from prem.gateway.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class CloudflareConnector(BaseConnector):
    def __init__(self, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/"

    def get_headers(self):
        return {"Authorization": f"Bearer {settings.CLOUDFLARE_API_KEY}"}

    def parse_chunk(self, chunk):
        try:
            json_part = chunk.decode("utf-8").split(":", 1)[1].strip()
            parsed_json = json.loads(json_part)
            value = parsed_json["response"]
        except Exception:
            value = ""
        return {
            "id": None,
            "model": None,
            "object": None,
            "created": str(timezone.now()),
            "choices": [
                {
                    "delta": {"content": value, "role": "assistant"},
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

        input = {"messages": messages, "stream": stream}
        try:
            response = requests.post(f"{self.base_url}{model}", headers=self.get_headers(), json=input, stream=True)
        except Exception as error:
            raise exceptions.PremProviderAPIConnectionError(
                error, provider="cloudflare", model=model, provider_message=str(error)
            )

        if response.status_code != 200:
            if response.status_code in self.status_code_exception_mapping:
                raise self.status_code_exception_mapping[response.status_code](
                    response.text, provider="cloudflare", model=model, provider_message=str(response.text)
                )
            else:
                logger.error(f"Unknown error from Cloudflare: {response.text} ({response.status_code})")
                raise exceptions.PremProviderError(
                    response.text, provider="cloudflare", model=model, provider_message=str(response.text)
                )

        if stream:
            return response

        plain_response = {
            "id": None,
            "choices": [
                {
                    "message": {"content": response.json()["result"]["response"], "role": "assistant"},
                }
            ],
            "created": str(timezone.now()),
            "model": model,
            "provider_name": "Cloudflare",
            "provider_id": "cloudflare",
        }
        return plain_response
