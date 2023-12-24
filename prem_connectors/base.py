from collections.abc import Sequence

import tiktoken

from prem.gateway import exceptions


class BaseConnector:
    def __init__(self, prompt_template: str = None):
        self.prompt_template = prompt_template
        self.status_code_exception_mapping = {
            403: exceptions.PremProviderPermissionDeniedError,
            422: exceptions.PremProviderUnprocessableEntityError,
            500: exceptions.PremProviderInternalServerError,
            401: exceptions.PremProviderAuthenticationError,
            400: exceptions.PremProviderBadRequestError,
            404: exceptions.PremProviderNotFoundError,
            429: exceptions.PremProviderRateLimitError,
            409: exceptions.PremProviderConflictError,
        }

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
