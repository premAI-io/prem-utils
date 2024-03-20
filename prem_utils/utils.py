import json
from datetime import datetime
from pathlib import Path

RELATIVE_MODELS_JSON_PATH = "./models.json"


def get_models_data() -> dict:
    models_json_path = Path(__file__).parent / RELATIVE_MODELS_JSON_PATH
    return json.loads(models_json_path.read_text())


def convert_timestamp(timestamp: str) -> str:
    try:
        return int(datetime.fromisoformat(timestamp).timestamp())
    except ValueError:
        return None
