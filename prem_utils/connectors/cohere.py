import json
from collections.abc import Sequence
from tempfile import NamedTemporaryFile
from uuid import uuid4

import cohere
from cohere.error import CohereAPIError, CohereConnectionError
from cohere.responses import Dataset

from prem_utils import errors
from prem_utils.connectors.base import BaseConnector


class CohereConnector(BaseConnector):
    def __init__(self, api_key: str, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = cohere.Client(api_key)
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
        chat_history, message = self.preprocess_messages(messages)
        try:
            response = self.client.chat(
                chat_history=chat_history,
                message=message,
                model="command",
                temperature=temperature,
                stream=stream,
            )
        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), errors.PremProviderError)
            raise custom_exception(error, provider="cohere", model=model, provider_message=str(error))

        if stream:
            return response
        plain_response = {
            "id": response.id,
            "choices": [
                {
                    "finish_reason": None,
                    "index": None,
                    "message": {"content": response.text, "role": "assistant"},
                }
            ],
            "created": None,
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
            "usage": None,
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }

    def finetuning(
        self,
        training_data: list[dict],
        validation_data: list[dict] | None = None,
        num_epochs: int = 3,
        model: str | None = None,
    ) -> str:
        id = uuid4().hex
        dataset = self._create_dataset(training_data, validation_data)
        response = self.client.create_custom_model(
            name=f"ft-model-{id}",
            dataset=dataset,
            model_type="CHAT",
            hyperparameters={"train_epochs": num_epochs} if num_epochs else None,
        )
        return response.id

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        response = self.client.get_custom_model(job_id)
        return {
            "id": response.id,
            "fine_tuned_model": response.model_id,
            "created_at": response.created_at,
            "finished_at": response.completed_at,
            "status": response.status,
            "error": None,
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }

    def _create_dataset(self, training_data: list[dict], validation_data: list[dict] | None = None) -> Dataset:
        id = uuid4().hex
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

    def _transform_data(self, data: list[dict]):
        return [
            {
                "messages": [
                    {
                        "role": "System",
                        "content": "You are a writing assistant that helps the user write coherent text.",
                    },
                    {"role": "User", "content": row["input"]},
                    {"role": "Chatbot", "content": row["output"]},
                ]
            }
            for row in data
        ]
