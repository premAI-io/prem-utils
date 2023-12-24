import json
import logging

import requests

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class CloudflareConnector(BaseConnector):
    def __init__(self, account_id: str, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        self.api_key = api_key

    def get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

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
            "created": None,
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
            response = requests.post(
                f"{self.base_url}{model}",
                headers=self.get_headers(),
                json=input,
                stream=True,
            )
        except Exception as error:
            raise errors.PremProviderAPIConnectionError(
                error, provider="cloudflare", model=model, provider_message=str(error)
            )

        if response.status_code != 200:
            if response.status_code in self.status_code_exception_mapping:
                raise self.status_code_exception_mapping[response.status_code](
                    response.text,
                    provider="cloudflare",
                    model=model,
                    provider_message=str(response.text),
                )
            else:
                logger.error(f"Unknown error from Cloudflare: {response.text} ({response.status_code})")
                raise errors.PremProviderError(
                    response.text,
                    provider="cloudflare",
                    model=model,
                    provider_message=str(response.text),
                )

        if stream:
            return response

        plain_response = {
            "id": None,
            "choices": [
                {
                    "message": {
                        "content": response.json()["result"]["response"],
                        "role": "assistant",
                    },
                }
            ],
            "created": None,
            "model": model,
            "provider_name": "Cloudflare",
            "provider_id": "cloudflare",
        }
        return plain_response
