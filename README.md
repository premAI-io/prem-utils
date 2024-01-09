# 🛠️ Prem Utils

Utilities, and Connectors in order to interact with all Model Serving and Fine-tuning Providers.

## 🤙 Usage

```bash
pip install prem-utils
```

```python
from prem_utils.connectors import openai

connector = openai.OpenAIConnector(api_key="")

prompt = "Hello, how are you?"
response = connector.chat_completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

message = response["choices"][0]["message"]["content"]
print(message)
```
