import json
import os
import tempfile
import time

import requests
from replicate import Client
from replicate.exceptions import ModelError, ReplicateError

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class ReplicateConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = Client(api_token=api_key)
        self.api_key = api_key
        self.exception_mapping = {
            ReplicateError: errors.PremProviderInternalServerError,
            ModelError: errors.PremProviderAPIStatusError,
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
                response = self.client.stream(
                    ref=model,
                    input={"prompt": prompt},
                )
                return response
            else:
                content = ""
                response = self.client.run(
                    ref=model,
                    input={"prompt": prompt},
                )
                content = "".join([element for element in response])
        except (ReplicateError, ModelError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="replicate", model=model, provider_message=str(error))

        plain_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": content, "role": "assistant"},
                }
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": model,
            "provider_name": "Replicate",
            "provider_id": "replicate",
            "usage": connector_utils.default_chatcompletions_usage(prompt, content),
        }
        return plain_response

    def finetuning(self, model: str, training_data: dict, validation_data: dict | None = None, num_epochs: int = 3):
        training_url = self._upload_and_transform_data(training_data, n_samples=25)
        model_name = f"{model.split(':')[0].split('/')[1]}-{int(time.time())}"
        fine_tuned_model = self.client.models.create(
            owner="filopedraz",
            name=model_name,
            visibility="private",
            hardware="cpu",
        )
        if validation_data:
            validation_url = self._upload_and_transform_data(validation_data, n_samples=5)
        else:
            validation_url = None
        training = self.client.trainings.create(
            version=model,
            input={
                "train_data": training_url,
                "validation_data": validation_url,
                "num_train_epochs": num_epochs,
                "batch_size_training": 1,
                "val_batch_size": 1,
            },
            destination=f"{fine_tuned_model.owner}/{fine_tuned_model.name}",
        )
        return training.id

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        response = self.client.trainings.get(job_id)
        return {
            "id": response.id,
            "fine_tuned_model": response.output["version"] if response.output else None,
            "created_at": response.created_at,
            "finished_at": response.completed_at,
            "status": response.status,
            "error": response.error,
            "provider_name": "Replicate",
            "provider_id": "replicate",
        }

    def _upload_and_transform_data(self, data, n_samples):
        file_name = f"dataset-{int(time.time())}.jsonl"
        url = f"https://dreambooth-api-experimental.replicate.com/v1/upload/{file_name}"

        headers = {"Authorization": f"Token {self.api_key}"}
        response = requests.post(url, headers=headers)
        response_data = response.json()
        upload_url = response_data.get("upload_url")
        serving_url = response_data.get("serving_url")

        if len(data) < n_samples:
            raise ValueError(f"Input 'data' must contain at least {n_samples} samples.")
        if not all(isinstance(row, dict) and {"input", "output"}.issubset(row.keys()) for row in data):
            raise ValueError("Input 'data' must be a list of dictionaries with 'input' and 'output' keys.")

        transformed_data = [
            {
                "prompt": row["input"],
                "completion": row["output"],
            }
            for row in data
        ]
        data_to_upload = [json.dumps(entry) for entry in transformed_data]
        jsonl_content = "\n".join(data_to_upload)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write(jsonl_content)

        try:
            with open(temp_file.name, "rb") as file:
                upload_response = requests.put(upload_url, data=file, headers={"Content-Type": "application/jsonl"})

            upload_response.raise_for_status()
        finally:
            os.remove(temp_file.name)

        return serving_url

    def embeddings(
        self,
        model: str,
        input: str,
        encoding_format: str = "float",
        user: str = None,
    ):
        if type(input) is not str:
            raise ValueError("Input 'input' must be a string.")
        try:
            response = self.client.run(
                ref=model,
                input={"text": input},
            )
            return {
                "data": [{"index": index, "embedding": data["embedding"]} for index, data in enumerate(response)],
                "model": model,
                "provider_name": "Replicate",
                "provider_id": "replicate",
                "usage": connector_utils.default_embeddings_usage(input),
            }
        except (ReplicateError, ModelError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="replicate", model=model, provider_message=str(error))
