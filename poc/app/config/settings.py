from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(
        default=None, description="API key for OpenAI services."
    )
    OPENAI_SOFT_MODEL: str = Field(
        default="gpt-5-mini", description="Chat model used for textual agents."
    )
    OPENAI_HARD_MODEL: str = Field(
        default="gpt-5", description="Chat model used for textual agents."
    )

    LOG_LEVEL: str = Field(default="DEBUG", description="Application log level.")

    PATH_DATA: Path = Field(default=Path().resolve() / "data")

    PATH_MAPA_ESTRUTURADO: Path = Field(
        default=Path().resolve() / "data" / "estruturado.yaml"
    )
    PATH_MAPA_NAO_ESTRUTURADO: Path = Field(
        default=Path().resolve() / "data" / "nao_estruturado.yaml"
    )
    PATH_SQLITE_DB: Path = Field(default=Path().resolve() / "data" / "dados.db")
    PATH_CHROMA_DB: Path = Field(default=Path().resolve() / "data" / "chroma_db")

    PATH_EVAL_FILE: Path = Field(default=Path().resolve() / "data" / "evaluation.json")

    model_config = SettingsConfigDict(
        env_file=".env",  # lê .env automaticamente
        env_file_encoding="utf-8",
        extra="ignore",  # ignora chaves extras
        case_sensitive=False,  # nomes de env não diferenciam maiúsc/minúsc
    )


load_dotenv()
SETTINGS = Settings()
