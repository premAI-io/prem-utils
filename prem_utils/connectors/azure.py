import json
import tempfile
import time
from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class AzureOpenAIConnector(BaseConnector):
    def __init__(self, api_key: str, base_url: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version="2023-10-01-preview",
        )
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
        try:
            response = self.client.chat.completions.create(
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
            raise custom_exception(error, provider="azure", model=model, provider_message=str(error))

        if stream:
            return response
        plain_response = {
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                }
                for choice in response.choices
            ],
            "created": response.created,
            "model": response.model,
            "provider_name": "Azure OpenAI",
            "provider_id": "azure_openai",
            "usage": connector_utils.default_chatcompletions_usage(
                messages, [choice.message.content for choice in response.choices]
            ),
        }
        return plain_response

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        try:
            response = self.client.embeddings.create(model=model, input=input)
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
            raise custom_exception(error, provider="azure", model=model, provider_message=str(error))
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
            "provider_name": "Azure OpenAI",
            "provider_id": "azure_openai",
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

        if not self._check_file_status(training_file_id):
            raise ValueError(f"Training file {training_file_id} is not ready.")

        if not self._check_file_status(validation_file_id):
            raise ValueError(f"Validation file {validation_file_id} is not ready.")

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
            raise custom_exception(error, provider="azure", model=model, provider_message=str(error))

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
            "provider_name": "Azure OpenAI",
            "provider_id": "azure_openai",
        }

    def _upload_and_transform_data(self, data: list[dict], size: int) -> str:
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
                raise custom_exception(error, provider="azure", model=None, provider_message=str(error))
        return response.id

    def _check_file_status(self, file_id: str) -> str:
        status = None
        while status not in ("processed", "error"):
            try:
                response = self.client.files.retrieve(file_id)
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
                raise custom_exception(error, provider="azure", model=None, provider_message=str(error))
            status = response.status
            time.sleep(0.5)
        return response.status == "processed"
