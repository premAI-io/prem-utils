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
from tabulate import tabulate as tab

from prem_utils import errors
from prem_utils.connectors.base import OpenAIBaseConnector, get_provider_blob


class OpenAIConnector(OpenAIBaseConnector):
    def __init__(self, api_key: str = None, base_url: str = None, prompt_template: str = None):
        super().__init__(api_key=api_key, base_url=base_url, prompt_template=prompt_template)
        self._errors = (
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
        )

    def list_models(self):
        provider_blob = get_provider_blob("openai")
        models = provider_blob["models"]

        # Convert models data into a list of lists for tabulate
        table_data = [[model.get(key, "") for key in models[0].keys()] for model in models]

        # Insert a header row with the keys of the models dictionary
        table_data.insert(0, models[0].keys())
        print(tab(table_data, headers="firstrow", tablefmt="grid"))

    def parse_chunk(self, chunk):
        return {
            "id": chunk.id,
            "model": chunk.model,
            "object": chunk.object,
            "created": chunk.created,
            "choices": [
                {
                    "delta": {
                        "content": choice.delta.content,
                        "role": choice.delta.role,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in chunk.choices
            ],
        }

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        try:
            response = self.client.embeddings.create(model=model, input=input)
        except self._errors as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))
        return {
            "data": [
                {
                    "index": embedding.index,
                    "embedding": embedding.embedding,
                }
                for embedding in response.data
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "provider_name": "OpenAI",
            "provider_id": "openai",
        }

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
        except self._errors as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))
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
            "provider_name": "OpenAI",
            "provider_id": "openai",
        }

    def _upload_and_transform_data(self, data: list[dict], size: int) -> str:
        """
        Transform and upload data to OpenAI in the required format.

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
            except self._errors as error:
                custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
                raise custom_exception(error, provider="openai", model=None, provider_message=str(error))
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
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                style=style,
                response_format=response_format,
            )
        except self._errors as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))
        return {"created": None, "data": [{"url": image.url, "b64_json": image.b64_json} for image in response.data]}
