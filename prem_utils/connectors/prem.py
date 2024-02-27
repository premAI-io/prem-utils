import uuid
from collections.abc import Generator
from datetime import datetime

import requests

from prem_utils.connectors.base import BaseConnector


class PremConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str | None = None) -> None:
        super().__init__(prompt_template=prompt_template)
        self.url_mappings = {
            "mamba": {
                "generation": "https://premai-io--generate-mamba.modal.run",
                "completion": "https://premai-io--completion-mamba.modal.run",
            },
            "phi2": {
                "generation": "https://premai-io--generate-phi2.modal.run",
                "completion": "https://premai-io--completion-phi2.modal.run",
            },
            "phi1-5": {
                "generation": "https://premai-io--generate-phi1-5.modal.run",
                "completion": "https://premai-io--completion-phi1-5.modal.run",
            },
            "stable_lm2": {
                "generation": "https://premai-io--generate-stable-lm2-zephyr.modal.run",
                "completion": "https://premai-io--completion-stable-lm2-zephyr.modal.run",
            },
            "tinyllama": {
                "generation": "https://premai-io--generate-tinyllama.modal.run",
                "completion": "https://premai-io--completion-tinyllama.modal.run",
            },
            "gemma": {
                "generation": "https://premai-io--generate-gemma.modal.run",
                "completion": "https://premai-io--completion-gemma.modal.run",
            },
        }
        self._api_key = api_key

    def parse_chunk(self, chunk):
        return {
            "id": chunk["id"],
            "model": chunk["model"],
            "object": chunk["object"],
            "created": chunk["created"],
            "choices": [
                {
                    "delta": {
                        "content": choice["delta"]["content"],
                        "role": choice["delta"]["role"],
                    },
                    "finish_reason": None,
                }
                for choice in chunk["choices"]
            ],
        }

    def _chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int | None = 128,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
    ):
        data = {"model": model, "temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}

        for message in messages:
            data["prompt"] = message["content"]
            try:
                response = requests.post(self.url_mappings[model]["completion"], json=data, timeout=600, stream=True)
                if response.status_code == 200:
                    for line in response.iter_lines():
                        token_to_sent = {
                            "id": uuid.uuid4().hex,
                            "model": model,
                            "object": "prem.chat_completion",
                            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "choices": [
                                {
                                    "delta": {"content": line.decode("utf-8"), "role": message["role"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield token_to_sent
                else:
                    yield {"status": response.status_code}
            except Exception as error:
                raise error

    def _chat_completion_generate(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int | None = 128,
        temperature: float | None = 1.0,
        top_p: float | None = 0.95,
    ) -> dict:
        data = {"model": model, "temperature": temperature, "max_new_tokens": max_tokens, "top_p": top_p}
        responses = []
        for message in messages:
            data["prompt"] = message["content"]
            try:
                response = requests.post(self.url_mappings[model]["generation"], json=data, timeout=600).json()
                responses.append(response)

            except Exception as e:
                responses.append({"status": 500, "error": str(e)})
        return responses

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str]],
        max_tokens: int | None = 128,
        frequency_penalty: float = 0,
        log_probs: int = None,
        logit_bias: dict[str, float] = None,
        presence_penalty: float = 0,
        seed: int | None = None,
        stop: str | list[str] = None,
        stream: bool = False,
        temperature: float = 1,
        top_p: float = 1,
    ) -> str | Generator[str, None, None]:
        try:
            if stream:
                return self._chat_completion_stream(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                return self._chat_completion_generate(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
        except Exception as error:
            raise error
