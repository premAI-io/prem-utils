import asyncio
import os

from dotenv import load_dotenv

from prem_utils.connectors.anthropic import AnthropicConnector
from prem_utils.connectors.anyscale import AnyscaleEndpointsConnector
from prem_utils.connectors.azure import AzureOpenAIConnector
from prem_utils.connectors.cohere import CohereConnector
from prem_utils.connectors.groq import GroqConnector
from prem_utils.connectors.mistral import MistralAzureConnector, MistralConnector
from prem_utils.connectors.openai import OpenAIConnector

load_dotenv()

messages = [{"role": "user", "content": "What is the weather like in San Francisco?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                },
                "required": ["location"],
            },
        },
    }
]
connectors = [
    {"connector": AnthropicConnector(api_key=os.environ["ANTHROPIC_API_KEY"]), "model": "claude-3-haiku-20240307"},
    {"connector": OpenAIConnector(api_key=os.environ["OPENAI_API_KEY"]), "model": "gpt-4o"},
    {
        "connector": AzureOpenAIConnector(
            api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=os.environ["AZURE_OPENAI_BASE_URL"]
        ),
        "model": "gpt-4-32k-azure",
    },
    {"connector": MistralConnector(api_key=os.environ["MISTRAL_AI_API_KEY"]), "model": "mistral-small-latest"},
    {
        "connector": MistralAzureConnector(
            api_key=os.environ["MISTRAL_AZURE_API_KEY"], endpoint=os.environ["MISTRAL_AZURE_ENDPOINT"]
        ),
        "model": "mistral-large",
    },
    {"connector": GroqConnector(api_key=os.environ["GROQ_API_KEY"]), "model": "groq/gemma-7b-it"},
    {"connector": CohereConnector(api_key=os.environ["COHERE_API_KEY"]), "model": "command-r-plus"},
    {
        "connector": AnyscaleEndpointsConnector(api_key=os.environ["ANYSCALE_API_KEY"]),
        "model": "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
    },
]

for connector_dict in connectors:
    connector = connector_dict["connector"]
    model = connector_dict["model"]
    print(f"Connector: {connector} and model: {model}")
    print("With tools")
    response = asyncio.run(connector.chat_completion(model=model, messages=messages, tools=tools))
    print(response)
    print("NO tools")
    response = asyncio.run(connector.chat_completion(model=model, messages=messages))
    print(response)
    print("Stream")
    if not isinstance(connector, CohereConnector):
        response = asyncio.run(connector.chat_completion(model=model, messages=messages, stream=True))
    else:
        response = connector.chat_completion(model=model, messages=messages, stream=True)
    print(response)
    print("\n", "-" * 50, "\n")
