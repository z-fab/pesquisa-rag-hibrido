import yaml
from config.settings import SETTINGS


def load_semantic_map_struct():
    with open(SETTINGS.PATH_MAPA_ESTRUTURADO, "r", encoding="utf-8") as f:
        struct_data = yaml.safe_load(f)
    return struct_data


def load_semantic_map_non_struct():
    with open(SETTINGS.PATH_MAPA_NAO_ESTRUTURADO, "r", encoding="utf-8") as f:
        non_struct_data = yaml.safe_load(f)
    return non_struct_data
