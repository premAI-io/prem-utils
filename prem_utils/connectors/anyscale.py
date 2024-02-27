from collections.abc import Sequence
from typing import Any

from prem_utils.connectors.openai import OpenAIConnector


class AnyscaleEndpointsConnector(OpenAIConnector):
    def __init__(
        self, api_key: str, base_url: str = "https://api.endpoints.anyscale.com/v1", prompt_template: str = None
    ) -> None:
        super().__init__(prompt_template=prompt_template, base_url=base_url, api_key=api_key)

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
        if "anyscale" in model:
            model = model.replace("anyscale/", "", 1)

        return super().chat_completion(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        if "anyscale" in model:
            model = model.replace("anyscale/", "", 1)

        return super().embeddings(model, input, encoding_format, user)

    def finetuning(
        self, model: str, training_data: list[dict], validation_data: list[dict] | None = None, num_epochs: int = 3
    ) -> str:
        return super().finetuning(model, training_data, validation_data, num_epochs)

    def get_finetuning_job(self, job_id) -> dict[str, Any]:
        return super().get_finetuning_job(job_id)

    def generate_image(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "standard",
        style: str = "vivid",
        response_format: str = "url",
    ):
        raise NotImplementedError
