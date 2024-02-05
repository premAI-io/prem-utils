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
    prem,
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
        elif connector["provider"] == "prem":
            connector_object = prem.PremConnector(api_key=os.environ["PREMAI_BEARER_TOKEN"])
        else:
            print(f"No connector for {connector['provider']}")

        text2text_models = [model for model in connector["models"] if model["model_type"] == "text2text"]
        text2vector_models = [model for model in connector["models"] if model["model_type"] == "text2vector"]
        text2image_models = [model for model in connector["models"] if model["model_type"] == "text2image"]

        if len(text2text_models) > 0:
            model_object = text2text_models[0]

            parameters = {}
            parameters["model"] = model_object["slug"]
            messages = [{"role": "user", "content": "Hello, how is it going?"}]
            messages.append({"role": "system", "content": "Behave like Rick Sanchez."})
            parameters["messages"] = messages

            print(f"Testing model {model_object['slug']} from {connector['provider']} connector \n\n\n")
            response = connector_object.chat_completion(stream=False, **parameters)
            print(f"\n\n\n Model {model_object['slug']} succeeed 🚀 \n\n\n")

            response = connector_object.chat_completion(stream=True, **parameters)
            for text in response:
                print(connector_object.parse_chunk(text))
            print(f"\n\n\n Model {model_object['slug']} succeeed with streaming 🚀 \n\n\n")

        if len(text2image_models) > 0:
            model_object = text2image_models[0]

            parameters = {}
            parameters["model"] = model_object["slug"]
            parameters["prompt"] = "A cute baby sea otter"
            parameters["size"] = "1024x1024"
            parameters["n"] = 1

            print(f"Testing model {model_object['slug']} from {connector['provider']} connector \n\n\n")
            response = connector_object.generate_image(**parameters)
            print(f"\n\n\n Model {model_object['slug']} succeeed 🚀 \n\n\n")

        if len(text2vector_models) > 0:
            model_object = text2vector_models[0]

            input = "Hello, how is it going?"
            print(f"Testing model {model_object['slug']} from {connector['provider']} connector")
            response = connector_object.embeddings(model=model_object["slug"], input=input)
            print(f"Embeddings: {len(response['data'][0])}")
            print(f"Model {model_object['slug']} succeeed 🚀 \n\n\n")


main()
