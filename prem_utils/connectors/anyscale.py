import json
import tempfile
import time
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
from openai.types.fine_tuning import FineTuningJob

from prem_utils import errors
from prem_utils.connectors.openai import OpenAIConnector
from prem_utils.types import Datapoint
from prem_utils.utils import convert_timestamp


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

    def create_job(
        self,
        model: str,
        training_dataset: list[Datapoint],
        validation_dataset: list[Datapoint] | None = None,
        num_epochs: int = 3,
    ) -> str:
        training_file_id = self._upload_data(training_dataset, size=20)

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
            raise custom_exception(error, provider="anyscale", model=None, provider_message=str(error))
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
            raise custom_exception(error, provider="anyscale", model=None, provider_message=str(error))
        return jobs

    def delete_job(self, id: str) -> None:
        self.client.fine_tuning.jobs.cancel(id)

    def _get_finetuning_job(self, job: FineTuningJob) -> dict[str, any]:
        return {
            "id": job.id,
            "model": job.fine_tuned_model,
            "created_at": convert_timestamp(str(job.created_at)),
            "finished_at": convert_timestamp(str(job.finished_at)) if job.finished_at else None,
            "status": self._parse_job_status(job.status),
            "provider_name": "Anyscale",
            "provider_id": "anyscale",
        }

    def _parse_job_status(self, status: str) -> str:
        if status == "pending":
            return "queued"
        elif status in ("running", "succeeded", "failed", "cancelled"):
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
                raise custom_exception(error, provider="anyscale", model=None, provider_message=str(error))
        return response.id

    def _check_file_status(self, file_id: str) -> str:
        status = None
        while status not in ("processed", "error"):
            print(status)
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
                raise custom_exception(error, provider="anyscale", model=None, provider_message=str(error))
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
        raise NotImplementedError
