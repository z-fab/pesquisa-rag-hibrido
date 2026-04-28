import yaml

from src.config.settings import SETTINGS


def load_structured_map() -> dict:
    """Loads the structured (SQL) semantic map from YAML."""
    with open(SETTINGS.PATH_STRUCTURED_MAP, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_unstructured_map() -> dict:
    """Loads the unstructured (documents) semantic map from YAML."""
    with open(SETTINGS.PATH_UNSTRUCTURED_MAP, encoding="utf-8") as f:
        return yaml.safe_load(f)
