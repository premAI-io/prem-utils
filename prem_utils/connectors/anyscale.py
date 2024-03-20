import json
import tempfile
from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from prem_utils import errors
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
        training_file_id = self._upload_and_transform_data(training_data, size=10)

        validation_file_id = None
        if validation_data:
            validation_file_id = self._upload_and_transform_data(validation_data, size=1)

        fine_tuning_params = {
            "model": model,
            "training_file": training_file_id,
            "hyperparameters": {"n_epochs": num_epochs},
        }

        if validation_file_id:
            fine_tuning_params["validation_file"] = validation_file_id

        try:
            response = self.client.fine_tuning.jobs.create(**fine_tuning_params)
        except (
            NotFoundError,
            APIResponseValidationError,
            ConflictError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
            BadRequestError,
            APIConnectionError,
            AuthenticationError,
            InternalServerError,
            PermissionDeniedError,
            UnprocessableEntityError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="anyscale", model=model, provider_message=str(error))
        return response.id

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        response = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": response.id,
            "fine_tuned_model": response.fine_tuned_model,
            "created_at": response.created_at,
            "finished_at": response.finished_at,
            "status": response.status,
            "error": response.error,
            "provider_name": "AnyScale",
            "provider_id": "anyscale",
        }

    def _upload_and_transform_data(self, data: list[dict], size: int) -> str:
        """
        Transform and upload data to AnyScale in the required format.

        Parameters:
        - data: The input data.

        Returns:
        - file_id: The ID of the uploaded file.
        """
        if len(data) < size:
            raise ValueError(f"Input 'data' must contain at least {size} rows.")
        if not all(isinstance(row, dict) and {"input", "output"}.issubset(row.keys()) for row in data):
            raise ValueError("Input 'data' must be a list of dictionaries with 'input' and 'output' keys.")

        transformed_data = [
            {
                "messages": [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
            }
            for row in data
        ]

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".jsonl", delete=False) as temp_file:
            for item in transformed_data:
                temp_file.write(json.dumps(item).encode("utf-8"))
                temp_file.write(b"\n")

            try:
                response = self.client.files.create(file=temp_file.file, purpose="fine-tune")
            except (
                NotFoundError,
                APIResponseValidationError,
                ConflictError,
                APIStatusError,
                APITimeoutError,
                RateLimitError,
                BadRequestError,
                APIConnectionError,
                AuthenticationError,
                InternalServerError,
                PermissionDeniedError,
                UnprocessableEntityError,
            ) as error:
                custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
                raise custom_exception(error, provider="anyscale", model=None, provider_message=str(error))
        return response.id

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
