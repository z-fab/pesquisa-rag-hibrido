from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from src.config.settings import SETTINGS, Settings


def get_llm(provider: str, model: str, settings: Settings) -> BaseChatModel:
    """Returns a configured LLM for the given provider and model."""
    match provider:
        case "openai":
            return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY, temperature=0)
        case "gemini":
            return ChatGoogleGenerativeAI(model=model, api_key=settings.GEMINI_API_KEY, temperature=0)
        case "ollama":
            return ChatOpenAI(
                model=model,
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",
                temperature=0,
            )
        case "groq":
            return ChatOpenAI(
                model=model,
                base_url=settings.GROQ_BASE_URL,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
            )
        case "openrouter":
            return ChatOpenAI(
                model=model,
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                temperature=0,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")


def _model_map(settings: Settings) -> dict[str, str]:
    return {
        "openai": settings.OPENAI_MODEL,
        "gemini": settings.GEMINI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
        "groq": settings.GROQ_MODEL,
        "openrouter": settings.OPENROUTER_MODEL,
    }


def get_inference_llm(settings: Settings) -> BaseChatModel:
    """Returns the inference LLM based on current settings."""
    provider = settings.PROVIDER
    return get_llm(provider, _model_map(settings)[provider], settings)


def get_node_llm(node_name: str, settings: Settings = SETTINGS) -> BaseChatModel:
    """Returns the LLM for a specific node with optional per-node override.

    Lê env vars `<NODE>_PROVIDER` e `<NODE>_MODEL` (uppercased) para permitir
    que cada node do grafo use um modelo diferente do global. Se nenhum
    override estiver definido, comportamento idêntico a `get_inference_llm()`.

    Chaves suportadas (convenção): PLANNER, SQL, SYNTHESIS, VERIFIER, ROUTER.
    Se apenas PROVIDER for setado sem MODEL, usa o modelo default daquele provider.
    Se apenas MODEL for setado sem PROVIDER, ignora (exige ambos para override).

    Exemplo:
        PLANNER_PROVIDER=openai PLANNER_MODEL=gpt-5 rag eval ...
    """
    import os

    key = node_name.upper()
    ov_provider = os.getenv(f"{key}_PROVIDER")
    ov_model = os.getenv(f"{key}_MODEL")

    if ov_provider:
        model = ov_model or _model_map(settings).get(ov_provider, "")
        if not model:
            raise ValueError(
                f"Node {node_name}: {key}_PROVIDER={ov_provider} setado mas "
                f"{key}_MODEL ausente e provider não tem default mapeado"
            )
        logger.debug(f"Node {node_name}: override provider={ov_provider} model={model}")
        return get_llm(ov_provider, model, settings)

    return get_inference_llm(settings)


def get_judge_llm(settings: Settings) -> BaseChatModel:
    """Returns the judge LLM based on current settings."""
    provider = settings.JUDGE_PROVIDER
    model = settings.JUDGE_MODEL or _model_map(settings).get(provider, "")
    return get_llm(provider, model, settings)


def maybe_with_structured_output(llm, schema, include_raw: bool = True):
    """Wraps an LLM with structured output only for providers that support json_schema.

    Groq, Ollama and some OpenRouter models reject `response_format: json_schema`
    at invocation time. For those, return the unwrapped LLM and let the calling
    node's fallback JSON parser handle the plain response.
    """
    supported = {p.strip() for p in SETTINGS.STRUCTURED_OUTPUT_PROVIDERS.split(",") if p.strip()}
    if SETTINGS.PROVIDER not in supported:
        logger.debug(f"Skipping structured output for provider={SETTINGS.PROVIDER} (not in {supported})")
        return llm
    try:
        return llm.with_structured_output(schema, include_raw=include_raw)
    except Exception as e:
        logger.warning(f"Failed to wrap LLM with structured output ({e}); falling back to plain LLM")
        return llm


def get_embeddings(settings: Settings) -> Embeddings:
    """Returns a configured embeddings model for the given provider and model."""
    match settings.EMBEDDING_PROVIDER:
        case "openai":
            return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
        case "gemini":
            return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.GEMINI_API_KEY)
        case "ollama":
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",
            )
        case _:
            raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")
