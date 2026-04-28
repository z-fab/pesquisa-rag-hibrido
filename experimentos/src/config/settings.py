from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Inference LLM
    PROVIDER: str = Field(default="openai")
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash")
    OLLAMA_MODEL: str = Field(default="llama3.1")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/v1")
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_MODEL: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")

    # Embeddings (independent)
    EMBEDDING_PROVIDER: str = Field(default="openai")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # Judge LLMs (independentes, até 3 juízes para triangulação)
    JUDGE_PROVIDER: str = Field(default="gemini")
    JUDGE_MODEL: str = Field(default="gemini-3.1-pro-preview")
    JUDGE_PROVIDER_2: str | None = Field(default=None)
    JUDGE_MODEL_2: str | None = Field(default=None)
    JUDGE_PROVIDER_3: str | None = Field(default=None)
    JUDGE_MODEL_3: str | None = Field(default=None)

    # Logging
    LOG_LEVEL: str = Field(default="DEBUG")

    # Structured-output capable providers (comma-separated).
    # Providers NOT in this list will skip `with_structured_output()` and rely
    # on fallback JSON parsing. Needed because Groq/Ollama only support
    # json_schema on specific models.
    STRUCTURED_OUTPUT_PROVIDERS: str = Field(default="openai,gemini")

    # SQL Node
    SQL_MAX_RETRIES: int = Field(default=3)

    # Text Retriever Node
    TEXT_SEARCH_K: int = Field(default=5)

    # Verifier / Graph
    VERIFIER_MAX_RETRIES: int = Field(default=2)

    # Ingest
    INGEST_CHUNK_SIZE: int = Field(default=1500)
    INGEST_CHUNK_OVERLAP: int = Field(default=200)

    # Paths
    PATH_DATA: Path = Field(default=_BASE_DIR / "data")
    PATH_STRUCTURED_MAP: Path = Field(default=_BASE_DIR / "data" / "structured.yaml")
    PATH_UNSTRUCTURED_MAP: Path = Field(default=_BASE_DIR / "data" / "unstructured.yaml")
    PATH_SQLITE_DB: Path = Field(default=_BASE_DIR / "data" / "dados.db")
    PATH_CHROMA_DB: Path = Field(default=_BASE_DIR / "data" / "chroma_db")
    PATH_EVAL_FILE: Path = Field(default=_BASE_DIR / "data" / "evaluation.json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


SETTINGS = Settings()
