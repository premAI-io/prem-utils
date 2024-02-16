from typing import Any

import requests

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class PremConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str | None = None) -> None:
        super().__init__(prompt_template=prompt_template)
        self._api_key = api_key

    def parse_chunk(self, chunk) -> dict[str, Any]:
        # Todo: Need to understand how it is used.
        pass

    def build_messages(self, messages: list[dict]) -> list[str]:
        # Todo: Whether it can be used in current providers
        pass

    def preprocess_messages(self, messages):
        # Todo: Need to understand whether to use it and how to use it.
        pass

    def chat_completion(
        self,
        model_name: str,
        messages: list[dict[str]],
        max_tokens: int,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
    ):
        assert model_name in ["phi-1-5", "phi-2", "tinyllama", "mamba-chat"], ValueError(
            "Models other than 'phi-1-5', 'phi-2', 'tinyllama', 'mamba-chat' are not supported"
        )

        # this is how msgs look like: [{'role': 'user', 'content': ...}]
        if model_name == "mamba-chat":
            _base_url = "https://mamba.compute.premai.io/v1/chat/completions"
            data = {
                "model": model_name,
                "messages": messages,
                "max_length": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            _base_url = f"https://{model_name}.compute.premai.io/mii/default"
            data = {
                "prompts": [message["content"] for message in messages],
                "max_length": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        _headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}

        try:
            response = requests.post(_base_url, json=data, headers=_headers)
        except (
            errors.PremProviderAPIErrror,
            errors.PremProviderPermissionDeniedError,
            errors.PremProviderUnprocessableEntityError,
            errors.PremProviderInternalServerError,
            errors.PremProviderAuthenticationError,
            errors.PremProviderBadRequestError,
            errors.PremProviderNotFoundError,
            errors.PremProviderRateLimitError,
            errors.PremProviderAPIResponseValidationError,
            errors.PremProviderConflictError,
            errors.PremProviderAPIStatusError,
            errors.PremProviderAPITimeoutError,
            errors.PremProviderAPIConnectionError,
        ) as error:
            raise error

        return response.text
