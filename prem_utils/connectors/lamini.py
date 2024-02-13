from collections.abc import Sequence

import lamini
from lamini import Embedding, Lamini
from lamini.error.error import (
    APIError,
    AuthenticationError,
    ModelNameError,
    RateLimitError,
    ServerTimeoutError,
    UnavailableResourceError,
    UserError,
)

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class LaminiConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        lamini.api_key = api_key
        self.exception_mapping = {
            "ModelNameError": errors.PremProviderError,
            "APIError": errors.PremProviderAPIErrror,
            "AuthenticationError": errors.PremProviderAuthenticationError,
            "RateLimitError": errors.PremProviderRateLimitError,
            "UserError": errors.PremProviderBadRequestError,
            "UnavailableResourceError": errors.PremProviderError,
            "ServerTimeoutError": errors.PremProviderAPITimeoutError,
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
        if stream:
            raise errors.PremProviderAPIErrror(
                provider="LaMini", message="Stream is not supported by LaMini", model=model
            )
        try:
            llm = Lamini(model_name=model)
            prompt = messages[-1]["content"]
            output = llm.generate(
                prompt=prompt,
                model_name=model,
                max_tokens=max_tokens,
                stop_tokens=stop,
            )
        except (
            ModelNameError,
            APIError,
            AuthenticationError,
            RateLimitError,
            UserError,
            UnavailableResourceError,
            ServerTimeoutError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="lamini", model=model, provider_message=str(error))
        plain_response = {
            "id": None,
            "choices": [
                {
                    "message": {
                        "content": output,
                        "role": "assistant",
                    },
                }
            ],
            "created": None,
            "model": model,
            "provider_name": "LaMini",
            "provider_id": "lamini",
        }
        return plain_response

    def finetuning(self, model: str, training_data: dict, validation_data: dict | None = None, num_epochs: int = 3):
        if not all(isinstance(row, dict) and {"input", "output"}.issubset(row.keys()) for row in training_data):
            raise ValueError("Input 'training_data' must be a list of dictionaries with 'input' and 'output' keys.")

        training_data = [{"input": row["input"], "output": row["output"]} for row in training_data]
        try:
            llm = Lamini(model_name=model)
            job = llm.train(data=training_data, is_public=False)
            return job["job_id"]
        except (
            ModelNameError,
            APIError,
            AuthenticationError,
            RateLimitError,
            UserError,
            UnavailableResourceError,
            ServerTimeoutError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="lamini", model=model, provider_message=str(error))

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        try:
            llm = Lamini(model_name=None)
            response = llm.check_job_status(job_id)
            status = str(response["status"]).lower()
            if status == "completed":
                status = "succeeded"
            return {
                "id": response["job_id"],
                "fine_tuned_model": response["model_name"],
                "created_at": response["start_time"],
                "finished_at": None,
                "status": status,
                "error": None,
                "provider_name": "LaMini",
                "provider_id": "lamini",
            }
        except (
            ModelNameError,
            APIError,
            AuthenticationError,
            RateLimitError,
            UserError,
            UnavailableResourceError,
            ServerTimeoutError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="lamini", model=None, provider_message=str(error))

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        try:
            embed = Embedding()
            embeddngs = [list(emb) for emb in embed.generate(input)]
            return {
                "data": [
                    {
                        "index": index,
                        "embedding": embedding,
                    }
                    for index, embedding in enumerate(embeddngs)
                ],
                "model": None,
                "usage": None,
                "provider_name": "LaMini",
                "provider_id": "lamini",
            }
        except (
            ModelNameError,
            APIError,
            AuthenticationError,
            RateLimitError,
            UserError,
            UnavailableResourceError,
            ServerTimeoutError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="lamini", model=None, provider_message=str(error))
