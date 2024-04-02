# 🛠️ Prem Utils

Utilities, and Connectors in order to interact with all Model Serving and Fine-tuning Providers.

## 🗄️ Connectors

| Name        | Status     |
| ----------- | ---------- |
| Anthropic   | Active     |
| OpenAI      | Active     |
| Perplexity  | Active     |
| Cohere      | Active     |
| DeepInfra   | Deprecated |
| FireworksAI | Deprecated |
| OctoAI      | Deprecated |
| Groq        | Active     |
| Lamini      | Deprecated |
| Mistral     | Active     |
| OpenRouter  | Active     |
| Prem        | Active     |
| Replicate   | Deprecated |
| Together    | Deprecated |

## 🤙 Usage

```bash
pip install prem-utils
```

```python
import asyncio
from inspect import iscoroutine
from prem_utils.connectors import openai

connector = openai.OpenAIConnector(api_key="")

prompt = "Hello, how are you?"
response = connector.chat_completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

if iscoroutine(response):
    response = asyncio.run(response)

message = response["choices"][0]["message"]["content"]
print(message)
```

## 📦 Contribute

### Install the necessary dependencies

```bash
virtualenv venv -p=3.11
source venv/bin/activate
pip install -r requirements.txt
```

### Test all or one connector

```bash
# will run all the connectors
python e2e.py

# only one connector
python e2e.py --name perplexity
```
