import json
import logging
import os
import pathlib

from dotenv import load_dotenv

from prem_utils.connectors import (
    anthropic,
    azure,
    cloudflare,
    cohere,
    fireworksai,
    mistral,
    octoai,
    openai,
    replicate,
    together,
)

logger = logging.getLogger(__name__)

load_dotenv()


def load_models_file():
    try:
        return json.loads(pathlib.Path("./prem_utils/models.json").read_text())
    except FileNotFoundError:
        logger.error("models.json file not found.")
        return None
    except json.JSONDecodeError:
        logger.error("models.json file contains invalid JSON.")
        return None


def main():
    models_file = load_models_file()

    for connector in models_file["connectors"]:
        if connector["provider"] == "anthropic":
            connector_object = anthropic.AnthropicConnector(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif connector["provider"] == "azure":
            connector_object = azure.AzureOpenAIConnector(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            )
        elif connector["provider"] == "cohere":
            connector_object = cohere.CohereConnector(api_key=os.environ["COHERE_API_KEY"])
        elif connector["provider"] == "fireworksai":
            connector_object = fireworksai.FireworksAIConnector(api_key=os.environ["FIREWORKS_AI_API_KEY"])
        elif connector["provider"] == "octoai":
            connector_object = octoai.OctoAIConnector(api_key=os.environ["OCTO_AI_API_KEY"])
        elif connector["provider"] == "openai":
            connector_object = openai.OpenAIConnector(api_key=os.environ["OPENAI_API_KEY"])
        elif connector["provider"] == "replicate":
            connector_object = replicate.ReplicateConnector(api_key=os.environ["REPLICATE_API_KEY"])
        elif connector["provider"] == "together":
            connector_object = together.TogetherConnector(api_key=os.environ["TOGETHER_AI_API_KEY"])
        elif connector["provider"] == "cloudflare":
            connector_object = cloudflare.CloudflareConnector(
                api_key=os.environ["CLOUDFLARE_API_KEY"],
                account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
            )
        elif connector["provider"] == "mistralai":
            connector_object = mistral.MistralConnector(api_key=os.environ["MISTRAL_AI_API_KEY"])
        elif connector["provider"] == "deepinfra":
            connector_object = openai.OpenAIConnector(
                api_key=os.environ["DEEP_INFRA_API_KEY"],
                base_url="https://api.deepinfra.com/v1/openai",
            )
        else:
            print(f"No connector for {connector['provider']}")

        model_object = connector["models"][0]
        if model_object["model_type"] == "text2text":
            parameters = {}
            parameters["model"] = model_object["slug"]
            messages = [{"role": "user", "content": "Hello, how is it going?"}]
            messages.append({"role": "system", "content": "Behave like Rick Sanchez."})
            parameters["messages"] = messages

            print(f"Testing model {model_object['slug']} from {connector['provider']} connector \n\n\n")
            response = connector_object.chat_completion(stream=False, **parameters)
            print(response)
            print(f"\n\n\n Model {model_object['slug']} succeeed 🚀 \n\n\n")

            response = connector_object.chat_completion(stream=True, **parameters)
            for text in response:
                print(connector_object.parse_chunk(text))
            print(f"\n\n\n Model {model_object['slug']} succeeed with streaming 🚀 \n\n\n")


main()
