import json
import httpx
import modal
import uuid
from collections.abc import Generator
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Any

import requests

from prem_utils.connectors.base import BaseConnector


class PremConnector(BaseConnector):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://premai-io--slm-chat-completion-dev.modal.run/",
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(prompt_template=prompt_template)
        self.base_url = base_url
        self._api_key = api_key

    def parse_chunk(self, chunk):
        return {
            "id": chunk["id"],
            "model": chunk["model"],
            "object": chunk["object"],
            "created": chunk["created"],
            "choices": [
                {
                    "delta": {
                        "content": choice["delta"]["content"],
                        "role": choice["delta"]["role"],
                    },
                    "finish_reason": None,
                }
                for choice in chunk["choices"]
            ],
        }

    def _chat_completion_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.99,
    ) -> Generator[Any, Any, Any]:
        data = {
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "prompt": prompt,
        }

        modal_model = modal.Cls.lookup("slm-modal", model)

        try:
            for response in modal_model.completion.remote_gen(data, stream=True):
                token_to_sent = {
                    "id": uuid.uuid4().hex,
                    "model": model,
                    "object": "prem.chat_completion",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "choices": [
                        {
                            "delta": {"content": response, "role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield json.dumps(token_to_sent) + "\n"
        except httpx.RequestError as error:
            yield {"status": error.response.status_code if error.response else None}
            raise error


    def _chat_completion_generate(
        self, model: str, prompt: str, max_tokens: int = 128, temperature: float = 1.0, top_p: float = 0.99
    ) -> dict:
        data = {
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "prompt": prompt,
        }

        modal_model = modal.Cls.lookup("slm-modal", model)
        try:
            # The modal function is a generator, so we need to iterate over it to get the response
            # Though if the stream is set to False, we can just get the first response (the only response)
            for response in modal_model.completion.remote_gen(data, stream=False):
                return response
        except httpx.RequestError as e:
            return {"status": e.response.status_code if e.response else 500, "error": str(e)}
        
    def chat_completion(
        self,
        model: str,
        messages: str,
        max_tokens: int | None = 128,
        stream: bool = False,
        temperature: float = 1,
        top_p: float = 0.99,
    ) -> str | Generator[str, None, None]:
        try:
            if stream:
                return self._chat_completion_stream(
                    model=model,
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                print(model, messages, max_tokens, temperature, top_p)
                return self._chat_completion_generate(
                    model=model,
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
        except Exception as error:
            raise error

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
