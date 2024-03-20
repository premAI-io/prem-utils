import argparse
import json
import logging
import os
import pathlib

from dotenv import load_dotenv

from prem_utils.connectors import (
    anthropic,
    anyscale,
    azure,
    cloudflare,
    cohere,
    deepinfra,
    fireworksai,
    groq,
    mistral,
    octoai,
    openai,
    openrouter,
    perplexity,
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


def run_single_connector(connector_name: str) -> None:
    connectors_mapping = load_models_file()["connectors"]

    for conn in connectors_mapping:
        if conn["provider"] == connector_name:
            connector = conn

    connector_class_mapping = {
        "anthropic": (anthropic.AnthropicConnector, "ANTHROPIC_API_KEY"),
        "azure": (azure.AzureOpenAIConnector, "AZURE_OPENAI_API_KEY"),
        "cohere": (cohere.CohereConnector, "COHERE_API_KEY"),
        "fireworksai": (fireworksai.FireworksAIConnector, "FIREWORKS_AI_API_KEY"),
        "octoai": (octoai.OctoAIConnector, "OCTO_AI_API_KEY"),
        "openai": (openai.OpenAIConnector, "OPENAI_API_KEY"),
        "replicate": (replicate.ReplicateConnector, "REPLICATE_API_KEY"),
        "together": (together.TogetherConnector, "TOGETHER_AI_API_KEY"),
        "cloudflare": (cloudflare.CloudflareConnector, "CLOUDFLARE_API_KEY"),
        "mistralai": (mistral.MistralConnector, "MISTRAL_AI_API_KEY"),
        "prem": (prem.PremConnector, "PREM_AI_API_KEY"),
        "deepinfra": (deepinfra.DeepInfraConnector, "DEEP_INFRA_API_KEY"),
        "perplexity": (perplexity.PerplexityAIConnector, "PERPLEXITY_API_KEY"),
        "anyscale": (anyscale.AnyscaleEndpointsConnector, "ANYSCALE_API_KEY"),
        "openrouter": (openrouter.OpenRouterConnector, "OPENROUTER_API_KEY"),
        "groq": (groq.GroqConnector, "GROQ_API_KEY"),
    }

    if connector_name == "cloudflare":
        connector_object = connector_class_mapping[connector_name][0](
            api_key=os.environ[connector_class_mapping[connector_name][1]],
            account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        )
    elif connector_name == "azure":
        connector_object = connector_class_mapping[connector_name][0](
            api_key=os.environ[connector_class_mapping[connector_name][1]],
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
        )
    else:
        connector_object = connector_class_mapping[connector_name][0](
            api_key=os.environ[connector_class_mapping[connector_name][1]]
        )

    text2text_models = [model for model in connector["models"] if model["model_type"] == "text2text"]
    text2vector_models = [model for model in connector["models"] if model["model_type"] == "text2vector"]
    text2image_models = [model for model in connector["models"] if model["model_type"] == "text2image"]

    if len(text2text_models) > 0:
        model_object = text2text_models[0]

        parameters = {}
        parameters["model"] = model_object["slug"]

        messages = [{"role": "system", "content": "Behave like Rick Sanchez."}]
        messages.append({"role": "user", "content": "Hello, how is it going?"})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing different providers from prem-utils package")

    parser.add_argument(
        "--name",
        help=(
            "The following providers are supported:\n"
            "openai, azure, anthropic, cloudflare, cohere, fireworksai, lamini, mistral, ocotoai, deepinfra, prem, replicate, together\n"  # noqa: E501
            "You can choose any one of them to test it out. Please note: you should include the provider's API key in the .env file. You can check .env.template for your reference. if you put 'all' then all the providers will be used at once"  # noqa: E501
        ),
        default="all",
    )

    args = parser.parse_args()

    if args.name == "all":
        connectors_mapping = load_models_file()["connectors"]
        connector_name_list = [connector["provider"] for connector in connectors_mapping]

        for connector in connector_name_list:
            logger.info("=" * 20, f"Running for Connector: {connector}", "=" * 20)
            run_single_connector(connector_name=connector)
    else:
        run_single_connector(connector_name=args.name)
