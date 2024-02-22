import json
from collections.abc import Sequence

import tiktoken
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
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

from prem_utils import errors


def get_provider_blob(provider_name: str) -> dict:
    try:
        with open("./prem_utils/models.json") as file:
            data = json.load(file)
            for connector in data["connectors"]:
                if connector["provider"] == provider_name:
                    return connector
            print(f"No data found for provider: {provider_name}")
            return None
    except FileNotFoundError:
        print("JSON file not found.")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return None


class BaseConnector:
    def __init__(self, prompt_template: str = None):
        self.prompt_template = prompt_template
        self.status_code_exception_mapping = {
            403: errors.PremProviderPermissionDeniedError,
            422: errors.PremProviderUnprocessableEntityError,
            500: errors.PremProviderInternalServerError,
            401: errors.PremProviderAuthenticationError,
            400: errors.PremProviderBadRequestError,
            404: errors.PremProviderNotFoundError,
            429: errors.PremProviderRateLimitError,
            409: errors.PremProviderConflictError,
        }

    def list_models(self):
        raise NotImplementedError

    def apply_prompt_template(self, messages):
        prompt = self.prompt_template
        system_prompt = ""
        user_prompt = messages[-1]["content"]
        for message in messages:
            if message["role"] == "system":
                system_prompt = f"{system_prompt} {message['content']}"
        prompt = prompt.format(system_prompt=system_prompt, user_prompt=user_prompt)
        return prompt

    def chat_completion(self):
        raise NotImplementedError

    def get_number_of_tokens_request(self, messages: list):
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_number_of_tokens_response(self, text: str):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def embeddings(
        self,
        model: str,
        input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        encoding_format: str = "float",
        user: str = None,
    ):
        return {
            "data": None,
            "model": None,
            "usage": None,
            "provider_name": "Cohere",
            "provider_id": "cohere",
        }

    def generate_image(self):
        raise NotImplementedError

    def finetuning(
        self, model: str, training_data: list[dict], validation_data: list[dict] | None = None, num_epochs: int = 3
    ) -> str:
        raise NotImplementedError

    def get_finetuning_job(self, job_id) -> dict[str, any]:
        raise NotImplementedError


# The reason we have a base class for OpenAI is because this can easily integrate
# providers which provides services through Open AI's SDK
class OpenAIBaseConnector(BaseConnector):
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
        else:
            self.client = OpenAI(api_key=api_key)

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
        if self.prompt_template is not None:
            messages = self.apply_prompt_template(messages)

        # NOTE custom logic for providers who don't have
        # their sdk, but they use direclty OpenAI python client.
        if "deepinfra" in model:
            model = model.replace("deepinfra/", "", 1)
            max_tokens = max_tokens or 1024

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
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
            raise custom_exception(error, provider="openai", model=model, provider_message=str(error))

        if stream:
            return response
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
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in choice.message.tool_calls or []
                        ],
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
