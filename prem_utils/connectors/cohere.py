import json
from collections.abc import Sequence
from tempfile import NamedTemporaryFile
from uuid import uuid4

import cohere
from cohere.error import CohereAPIError, CohereConnectionError
from cohere.responses import Dataset
from cohere.responses.custom_model import CustomModel

from prem_utils import errors
from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector
from prem_utils.types import Datapoint

ROLE_MAPPING = {
    "user": "User",
    "assistant": "Chatbot",
    "system": "System",
}


class CohereConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = cohere.Client(api_key)
        self.async_client = cohere.AsyncClient(api_key)
        self.exception_mapping = {
            CohereAPIError: errors.PremProviderAPIErrror,
            CohereConnectionError: errors.PremProviderAPIConnectionError,
        }

    def preprocess_messages(self, messages):
        chat_history = []
        user_messages = []
        system_prompt = []
        for message in messages:
            if message["role"] == "user":
                user_messages.append(message["content"])
                chat_history.append({"user_name": "User", "text": message["content"]})
            elif message["role"] == "assistant":
                chat_history.append({"user_name": "Chatbot", "text": message["content"]})
            elif message["role"] == "system":
                system_prompt.append({"user_name": "System", "text": message["content"]})
        return system_prompt + chat_history, user_messages[-1]

    def parse_chunk(self, chunk):
        if hasattr(chunk, "text"):
            text = chunk.text
        else:
            text = ""
        return {
            "id": chunk.id,
            "model": None,
            "object": None,
            "created": None,
            "choices": [
                {
                    "delta": {"content": text, "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }

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
    ):
        chat_history, message = self.preprocess_messages(messages)
        try:
            if stream:
                return await self.async_client.chat(
                    chat_history=chat_history,
                    max_tokens=max_tokens,
                    message=message,
                    model=model,
                    p=top_p,
                    temperature=temperature,
                    stream=stream,
                )

            response = self.client.chat(
                chat_history=chat_history,
                max_tokens=max_tokens,
                message=message,
                model=model,
                p=top_p,
                temperature=temperature,
                stream=stream,
            )

        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="cohere", model=model, provider_message=str(error))

        plain_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": response.text, "role": "assistant"},
                }
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": model,
            "provider_name": "Cohere",
            "provider_id": "cohere",
            "usage": {
                "completion_tokens": response.token_count["prompt_tokens"],
                "prompt_tokens": response.token_count["response_tokens"],
                "total_tokens": response.token_count["total_tokens"],
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
            texts = input if isinstance(input, list) else [input]
            response = self.client.embed(texts=texts, model=model, input_type="search_document")
        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="cohere", model=model, provider_message=str(error))
        return {
            "data": [{"index": index, "embedding": embedding} for index, embedding in enumerate(response.embeddings)],
            "model": model,
            "usage": connector_utils.default_embeddings_usage(texts),
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }

    def create_job(
        self,
        training_dataset: list[Datapoint],
        model: str | None = None,
        validation_dataset: list[Datapoint] | None = None,
        num_epochs: int = 3,
    ) -> str:
        if model is not None:
            raise errors.PremProviderError(
                "Cohere does not support fine-tuning with a specific model. Please use the default model.",
                provider="cohere",
            )
        id = uuid4().hex
        dataset = self._create_dataset(training_dataset, validation_dataset)
        response = self.client.create_custom_model(
            name=f"ft-model-{id}",
            dataset=dataset,
            model_type="CHAT",
            hyperparameters={"train_epochs": num_epochs} if num_epochs else None,
        )
        return response.id

    def mget_jobs(self, ids: list[str]) -> list[dict[str, any]]:
        jobs = []
        for job_id in ids:
            model = self.client.get_custom_model(job_id)
            jobs.append(self._get_finetuning_job(model))
        return jobs

    def list_jobs(self) -> list[dict[str, any]]:
        return [self._get_finetuning_job(model) for model in self.client.list_custom_models()]

    def _get_finetuning_job(self, model: CustomModel) -> dict[str, any]:
        return {
            "id": model.id,
            "fine_tuned_model": model.model_id,
            "created_at": int(model.created_at.timestamp()),
            "finished_at": int(model.completed_at.timestamp()) if model.completed_at else None,
            "status": self._parse_job_status(model.status),
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }

    def _parse_job_status(self, status: str) -> str:
        if status == "QUEUED":
            return "queued"
        elif status == "PAUSED":
            return "cancelled"
        elif status == "FAILED":
            return "failed"
        elif status in ("CREATED", "DEPLOYING", "READY"):
            return "succeeded"
        elif status == "TRAINING":
            return "running"
        else:
            return "other"

    def _create_dataset(
        self, training_data: list[Datapoint], validation_data: list[Datapoint] | None = None
    ) -> Dataset:
        training_data = self._transform_data(training_data)
        training_data_file = NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        self._save_to_file(training_data, training_data_file.name)
        validation_data = self._transform_data(validation_data) if validation_data else None
        validation_data_file = NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") if validation_data else None
        if validation_data:
            self._save_to_file(validation_data, validation_data_file.name)

        dataset = self.client.create_dataset(
            name=f"dataset-{id}",
            dataset_type="chat-finetune-input",
            data=open(training_data_file.name, "rb"),
            eval_data=open(validation_data_file.name, "rb") if validation_data else None,
        )
        dataset.await_validation()
        training_data_file.close()
        if validation_data_file:
            validation_data_file.close()
        return dataset

    def _save_to_file(self, data: list[dict], file_name: str) -> None:
        with open(file_name, "w") as file:
            for item in data:
                json.dump(item, file)
                file.write("\n")

    def _transform_data(self, data: list[Datapoint]):
        return [
            {
                "messages": [
                    {"role": ROLE_MAPPING.get(message["role"]), "content": message["content"]}
                    for message in sample["messages"]
                ]
            }
            for sample in data
        ]
