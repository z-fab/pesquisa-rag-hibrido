import json

from config.settings import SETTINGS
from loguru import logger


def load_evaluation_json() -> dict:
    try:
        with open(SETTINGS.PATH_EVAL_FILE, "r", encoding="utf-8") as file:
            evaluation_data = json.load(file)
        return evaluation_data
    except FileNotFoundError:
        logger.warning(f"Evaluation file not found at {SETTINGS.PATH_EVAL_FILE}.")
        return {}
