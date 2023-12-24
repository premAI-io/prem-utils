import cohere
from cohere.error import CohereAPIError, CohereConnectionError
from django.conf import settings
from django.utils import timezone

from prem.gateway import exceptions
from prem.gateway.connectors.base import BaseConnector


class CohereConnector(BaseConnector):
    def __init__(self, prompt_template: str = None):
        super().__init__(prompt_template=prompt_template)
        self.client = cohere.Client(settings.COHERE_API_KEY)
        self.exception_mapping = {
            CohereAPIError: exceptions.PremProviderAPIErrror,
            CohereConnectionError: exceptions.PremProviderAPIConnectionError,
        }

    def preprocess_messages(self, messages):
        message = messages[-1]["content"]
        chat_history = []
        user_messages = []
        system_prompt = []
        for message in messages[:-1]:
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
            "created": str(timezone.now()),
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
                chat_history=chat_history, message=message, model="command", temperature=temperature, stream=stream
            )
        except (CohereAPIError, CohereConnectionError) as error:
            custom_exception = self.exception_mapping.get(type(error), exceptions.PremProviderError)
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
            "created": str(timezone.now()),
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
