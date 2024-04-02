import asyncio
import json
import queue
import threading
from collections.abc import Generator
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
import requests

from prem_utils.connectors.base import BaseConnector


class PremConnector(BaseConnector):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ml-development.prem.ninja/api/ml",
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(prompt_template=prompt_template)
        self.base_url = base_url
        self._api_key = api_key

    def parse_chunk(self, chunk):
        return {
            "id": chunk.get("id", ""),
            "model": chunk["model"],
            # "object": chunk["object"],
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

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = 128,
        frequency_penalty: float = 0,
        log_probs: int = None,
        logit_bias: dict[str, float] = None,
        presence_penalty: float = 0,
        seed: int | None = None,
        stop: str | list[str] = None,
        stream: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str | Generator[str, None, None]:
        # URL of your endpoint
        url = "http://localhost:8000/api/ml/slm-completion/"
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            # Add other parameters as needed
        }
        if stream:
            return self._stream_generator_wrapper(url, request_data)
        else:
            return self._perform_request(url, request_data)

    def _perform_request(self, url, request_data):
        request_headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        with httpx.Client() as client:
            response = client.post(url, json=request_data, headers=request_headers, timeout=300)
            return response.text

    def _stream_generator_wrapper(self, url, request_data):
        """
        Wraps the async streaming generator to be consumed like a synchronous generator.
        """
        # Queue to hold streamed chunks
        q = queue.Queue()

        def run_async():
            asyncio.run(self._consume_streaming_endpoint(url, request_data, q))

        # Start the asynchronous generator in a separate thread
        threading.Thread(target=run_async).start()

        # Yield from queue in the current (synchronous) thread
        while True:
            chunk = q.get()
            if chunk is None:  # None is used as a signal to indicate completion
                break
            yield chunk

    async def _consume_streaming_endpoint(self, url, request_data, q):
        """
        Asynchronous generator to consume a streaming endpoint and put the content into a queue.
        """
        request_headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=request_data, headers=request_headers, timeout=300) as response:
                async for chunk in response.aiter_text():
                    if len(chunk.strip()) != 0:
                        # Sometimes (randomly) chunks contains multiple lines
                        lines = chunk.strip().split("\n")
                        for line in lines:
                            q.put(json.loads(line))
        q.put(None)  # Signal completion

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
