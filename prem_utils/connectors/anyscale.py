import json
import tempfile

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class AnyScaleConnector(BaseConnector):
    def __init__(self, api_key: str = None, base_url: str = None, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.exception_mapping = {
            APIError: errors.PremProviderAPIErrror,
            PermissionDeniedError: errors.PremProviderPermissionDeniedError,
            UnprocessableEntityError: errors.PremProviderUnprocessableEntityError,
            InternalServerError: errors.PremProviderInternalServerError,
            AuthenticationError: errors.PremProviderAuthenticationError,
            BadRequestError: errors.PremProviderBadRequestError,
            NotFoundError: errors.PremProviderNotFoundError,
            RateLimitError: errors.PremProviderRateLimitError,
            APIResponseValidationError: errors.PremProviderAPIResponseValidationError,
            ConflictError: errors.PremProviderConflictError,
            APIStatusError: errors.PremProviderAPIStatusError,
            APITimeoutError: errors.PremProviderAPITimeoutError,
            APIConnectionError: errors.PremProviderAPIConnectionError,
        }
        self.client = OpenAI(api_key=api_key, base_url=base_url)

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
