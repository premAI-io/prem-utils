import json
import tempfile

import together
from together.error import AttributeError, AuthenticationError, InstanceError, JSONError, RateLimitError, ResponseError

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class TogetherConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        together.api_key = api_key
        self.exception_mapping = {
            AuthenticationError: errors.PremProviderAuthenticationError,
            ResponseError: errors.PremProviderResponseValidationError,
            JSONError: errors.PremProviderResponseValidationError,
            InstanceError: errors.PremProviderAPIStatusError,
            RateLimitError: errors.PremProviderRateLimitError,
            AttributeError: errors.PremProviderResponseValidationError,
        }

    def parse_chunk(self, chunk):
        return {
            "id": None,
            "model": None,
            "object": None,
            "created": None,
            "choices": [
                {
                    "delta": {"content": str(chunk), "role": "assistant"},
                    "finish_reason": None,
                }
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
        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        prompt = messages[-1]["content"]
        try:
            if stream:
                response = together.Complete.create_streaming(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=presence_penalty,
                )
                return response
            else:
                response = together.Complete.create(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=presence_penalty,
                )
                plain_response = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": index,
                            "message": {"content": choice["text"], "role": "assistant"},
                        }
                        for index, choice in enumerate(response["output"]["choices"])
                    ],
                    "created": connector_utils.default_chatcompletion_response_created(),
                    "model": model,
                    "provider_name": "Together",
                    "provider_id": "together",
                    "usage": connector_utils.default_chatcompletions_usage(
                        prompt, [choice["text"] for choice in response["output"]["choices"]]
                    ),
                }
                return plain_response
        except (
            AuthenticationError,
            ResponseError,
            JSONError,
            InstanceError,
            RateLimitError,
            AttributeError,
        ) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="together", model=model, provider_message=str(error))

    def finetuning(
        self, model: str, training_data: list[dict], validation_data: list[dict] | None = None, num_epochs: int = 3
    ) -> str:
        train_file_id = self._upload_and_transform_data(training_data, size=100)

        result = together.Finetune.create(model=model, training_file=train_file_id, n_epochs=num_epochs)
        return result["id"]

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        response = together.Finetune.retrieve(job_id)
        status = response["status"]
        if status == "error":
            status = "failed"
        elif status == "completed":
            status = "succeeded"
        return {
            "id": response["id"],
            "fine_tuned_model": response.get("model_output_name", None),
            "created_at": response["created_at"],
            "finished_at": response["updated_at"] if status in ("succeeded", "failed") else None,
            "status": status,
            "error": None,
            "provider_name": "Togheter",
            "provider_id": "togheter",
        }

    def _upload_and_transform_data(self, data: list[dict], size: int) -> str:
        if len(data) < size:
            raise ValueError(f"Input 'data' must contain at least {size} rows.")
        if not all(isinstance(row, dict) and {"input", "output"}.issubset(row.keys()) for row in data):
            raise ValueError("Input 'data' must be a list of dictionaries with 'input' and 'output' keys.")

        transformed_data = [{"text": f'{row["input"]} {row["output"]}'} for row in data]

        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".jsonl", delete=False) as temp_file:
            for item in transformed_data:
                temp_file.write(json.dumps(item).encode("utf-8"))
                temp_file.write(b"\n")

            try:
                upload_response = together.Files.upload(file=temp_file.name, check=False)
                return upload_response["id"]
            except (
                AuthenticationError,
                ResponseError,
                JSONError,
                InstanceError,
                RateLimitError,
                AttributeError,
            ) as error:
                custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
                raise custom_exception(error, provider="together", model=None, provider_message=str(error))
