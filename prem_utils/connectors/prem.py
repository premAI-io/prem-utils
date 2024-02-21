from collections.abc import Generator
from typing import Any

import requests

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector
from prem_utils.default_urls import DEFAULT_PREM_URLS


class PremConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str | None = None) -> None:
        super().__init__(prompt_template=prompt_template)
        self._api_key = api_key
        self._prem_errors = (
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
        )

    def parse_chunk(self, chunk) -> dict[str, Any]:
        pass

    def build_messages(self, messages: list[dict]) -> list[str]:
        pass

    def preprocess_messages(self, messages):
        # Todo: Need to understand whether to use it and how to use it.
        pass

    def _chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
    ):
        data = {"model": model, "temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}

        for message in messages:
            data["prompt"] = message["content"]
            try:
                response = requests.post(DEFAULT_PREM_URLS[model]["completion"], json=data, timeout=600, stream=True)
                if response.status_code == 200:
                    for line in response.iter_lines():
                        token_to_sent = {"status": 200, "content": line.decode("utf-8")}
                        yield token_to_sent
                else:
                    yield {"status": response.status_code}
            except self._prem_errors as error:
                raise error

    def _chat_completion_generate(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int,
        temperature: float | None = 1.0,
        top_p: float | None = 0.95,
    ) -> dict:
        data = {"model": model, "temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}
        responses = []
        for message in messages:
            data["prompt"] = message["content"]
            try:
                response = requests.post(DEFAULT_PREM_URLS[model]["generation"], json=data, timeout=600)
                responses.append(response.text)

            except self._prem_errors as e:
                responses.append({"status": 500, "error": str(e)})
        return responses

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        stream: bool | None = False,
    ) -> str | Generator[str, None, None]:
        assert model in ["phi1.5", "phi2", "tinyllama", "mamba", "stable_lm2"], ValueError(
            "Models other than 'phi1.5', 'phi2', 'tinyllama', 'mamba' , 'stable_lm2' are not supported"
        )

        if stream:
            return self._chat_completion_stream(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
        else:
            return self._chat_completion_generate(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
