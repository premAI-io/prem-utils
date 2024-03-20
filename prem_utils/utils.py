import json
from pathlib import Path

RELATIVE_MODELS_JSON_PATH = "./models.json"


def get_models_data() -> dict:
    models_json_path = Path(__file__).parent / RELATIVE_MODELS_JSON_PATH
    return json.loads(models_json_path.read_text())
