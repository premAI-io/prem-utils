import asyncio
import json
import os
import uuid
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
import requests

from prem_utils.connectors import utils as connector_utils
from prem_utils.connectors.base import BaseConnector


class PremConnector(BaseConnector):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ml.premai.io/api/ml",
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(prompt_template=prompt_template)
        self.base_url = base_url
        self._api_key = api_key
        self.prem_ml_key = os.environ["PREM_ML_API_KEY"]  # To authenticate prem_ml slm completion

    def parse_chunk(self, chunk):
        return {
            "id": chunk.get("id", str(uuid.uuid4())),
            "model": chunk["model"],
            "created": chunk["created"],
            "choices": [
                {
                    "delta": {
                        "content": choice["message"]["content"],
                        "role": choice["message"]["role"],
                    },
                    "finish_reason": None,
                }
                for choice in chunk["choices"]
            ],
        }

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = 128,
        stream: bool = False,
        temperature: float = 1,
        top_p: float = 0.99,
        frequency_penalty: float = 0,
        log_probs: int = None,
        logit_bias: dict[str, float] = None,
        presence_penalty: float = 0,
        seed: int | None = None,
        stop: str | list[str] = None,
    ) -> str | asyncio.streams.StreamReader:
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        prem_ml_url = f"{self.base_url}/slm-completion/"

        try:
            if stream:
                return self._stream_response(prem_ml_url, request_data)
            else:
                response = await self._perform_request(prem_ml_url, request_data)
                return self._format_response(messages, response)
        except Exception as e:
            raise e

    async def _perform_request(self, url, request_data):
        headers = {"Authorization": f"Bearer {self.prem_ml_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=request_data, headers=headers, timeout=600)
        return response.json()

    async def _stream_response(self, url, request_data):
        headers = {"Authorization": f"Bearer {self.prem_ml_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=request_data, headers=headers, timeout=600) as response:
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            yield json.loads(line)

    def _format_response(self, messages, response):
        plain_response = {
            "choices": [
                {
                    "finish_reason": str(choice["finish_reason"]),
                    "index": choice["index"],
                    "message": {
                        "content": choice["message"]["content"],
                        "role": choice["message"]["role"],
                    },
                }
                for choice in response["choices"]
            ],
            "created": connector_utils.default_chatcompletion_response_created(),
            "model": response["model"],
            "provider_name": "Prem",
            "provider_id": "premai",
            "usage": connector_utils.default_chatcompletions_usage(messages, response["choices"]),
        }
        return plain_response

    def _upload_data(self, data: list[dict]) -> str:
        try:
            temp_file = NamedTemporaryFile("w", suffix=".jsonl", delete=True)
            temp_file.writelines([f"{json.dumps(item)}\n" for item in data])
            temp_file.seek(0)
            files = {"file": open(temp_file.name, "rb")}
            response = requests.post(
                f"{self.base_url}/files/",
                files=files,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            file_id = response.json()["id"]
            temp_file.close()
            return file_id
        except Exception as error:
            raise error

    def finetuning(
        self,
        model: str,
        training_data: list[dict] | None = None,
        validation_data: list[dict] | None = None,
        hf_dataset: str | None = None,
        num_epochs: int = 3,
    ) -> str:
        if training_data and hf_dataset:
            raise ValueError("You can only provide either training_data or hf_dataset, not both")
        if validation_data and hf_dataset:
            raise ValueError("You can only provide either validation_data or hf_dataset, not both")
        training_file_id = self._upload_data(training_data) if training_data else None
        validation_data_id = self._upload_data(validation_data) if validation_data else None
        response = requests.post(
            f"{self.base_url}/fine-tuning/jobs/",
            json={
                "model": model,
                "training_file": training_file_id,
                "validation_file": validation_data_id,
                "hf_dataset": hf_dataset,
                "hyperparameters": {"num_epochs": num_epochs},
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return response.json()

    def get_finetuning_job(self, job_id) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/fine-tuning/jobs/{job_id}/",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return response.json()
