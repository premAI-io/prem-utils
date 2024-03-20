from collections.abc import Sequence
from typing import Any

from prem_utils.connectors.openai import OpenAIConnector


class DeepInfraConnector(OpenAIConnector):
    def __init__(
        self, api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", prompt_template: str = None
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
        if "deepinfra" in model:
            model = model.replace("deepinfra/", "", 1)

        if presence_penalty is None:
            presence_penalty = 0

        return super().chat_completion(
            model, messages, max_tokens, frequency_penalty, presence_penalty, seed, stop, stream, temperature, top_p
        )

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        raise NotImplementedError

    def finetuning(
        self, model: str, training_data: list[dict], validation_data: list[dict] | None = None, num_epochs: int = 3
    ) -> str:
        raise NotImplementedError

    def get_finetuning_job(self, job_id) -> dict[str, Any]:
        raise NotImplementedError

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
