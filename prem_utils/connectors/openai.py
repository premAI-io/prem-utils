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
    AsyncOpenAI,
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
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector
from prem_utils.types import Datapoint


class OpenAIConnector(BaseConnector):
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
        if base_url is not None:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)

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

    def _get_arguments(self, arguments):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return None

    async def chat_completion(
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
        tools=None,
    ):
        if tools is not None and stream:
            raise errors.PremProviderError(
                "Cannot use tools with stream=True",
                provider="openai",
                model=model,
                provider_message="Cannot use tools with stream=True",
            )

        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        request_data = dict(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens if max_tokens != 0 else None,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            logprobs=log_probs,
            logit_bias=logit_bias,
            tools=tools,
        )
        try:
            if stream:
                return await self.async_client.chat.completions.create(**request_data)

            response = self.client.chat.completions.create(**request_data)

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
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))

        plain_response = {
            "id": response.id,
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "function": {
                                    "arguments": self._get_arguments(tool_call.function.arguments),
                                    "name": tool_call.function.name,
                                },
                                "type": tool_call.type,
                            }
                            for tool_call in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None,
                    },
                }
                for choice in response.choices
            ],
            "created": response.created,
            "model": response.model,
            "provider_name": "OpenAI",
            "provider_id": "openai",
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
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

    def create_job(
        self,
        model: str,
        training_dataset: list[Datapoint],
        validation_dataset: list[Datapoint] | None = None,
        num_epochs: int = 3,
    ) -> str:
        training_file_id = self._upload_data(training_dataset, size=10)

        validation_file_id = None
        if validation_dataset:
            validation_file_id = self._upload_data(validation_dataset, size=1)

        fine_tuning_params = {
            "model": model,
            "training_file": training_file_id,
            "hyperparameters": {"n_epochs": num_epochs},
        }

        if validation_file_id:
            fine_tuning_params["validation_file"] = validation_file_id

        if not self._check_file_status(training_file_id):
            raise ValueError(f"Training file {training_file_id} is not ready.")

        if validation_file_id and not self._check_file_status(validation_file_id):
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
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))

        return response.id

    def mget_jobs(self, ids: list[str]) -> list[dict[str, any]]:
        jobs = []
        try:
            for job_id in ids:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                jobs.append(self._get_finetuning_job(job))
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
            raise custom_exception(error, provider="openai", model=None, provider_message=str(error))
        return jobs

    def list_jobs(self) -> list[dict[str, any]]:
        jobs = []
        try:
            response = self.client.fine_tuning.jobs.list()
            paginated_jobs = [self._get_finetuning_job(job) for job in response.data]
            last_job = paginated_jobs[-1]
            jobs.extend(paginated_jobs)
            while response.next_page_info():
                response = self.client.fine_tuning.jobs.list(after=last_job["id"])
                paginated_jobs = [self._get_finetuning_job(job) for job in response.data]
                if len(paginated_jobs) == 0:
                    break
                last_job = paginated_jobs[-1]
                jobs.extend(paginated_jobs)
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
            raise custom_exception(error, provider="openai", model=None, provider_message=str(error))
        return jobs

    def delete_job(self, id: str) -> None:
        self.client.fine_tuning.jobs.cancel(id)

    def _get_finetuning_job(self, job: FineTuningJob) -> dict[str, any]:
        return {
            "id": job.id,
            "model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "status": self._parse_job_status(job.status),
            "provider_name": "OpenAI",
            "provider_id": "openai",
        }

    def _parse_job_status(self, status: str) -> str:
        if status in ("validating_files", "running"):
            return "running"
        elif status in ("queued", "succeeded", "failed", "cancelled"):
            return status

    def _upload_data(self, data: list[Datapoint], size: int) -> str:
        if len(data) < size:
            raise ValueError(f"Input 'data' must contain at least {size} rows.")

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".jsonl", delete=False) as temp_file:
            for sample in data:
                temp_file.write(json.dumps(sample).encode("utf-8"))
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
                raise custom_exception(error, provider="openai", model=None, provider_message=str(error))
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
                raise custom_exception(error, provider="openai", model=None, provider_message=str(error))
            status = response.status
            time.sleep(0.5)
        return response.status == "processed"

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
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))
        return {"created": None, "data": [{"url": image.url, "b64_json": image.b64_json} for image in response.data]}
