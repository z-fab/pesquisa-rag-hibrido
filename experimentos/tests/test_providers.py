import pytest

from src.config.providers import get_inference_llm, get_judge_llm, get_llm, get_node_llm
from src.config.settings import Settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "OPENAI_API_KEY": "sk-openai",
        "GEMINI_API_KEY": "gk-gemini",
        "GROQ_API_KEY": "gk-groq",
        "OPENROUTER_API_KEY": "or-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_get_llm_groq_uses_groq_base_url_and_key():
    settings = _make_settings()
    llm = get_llm("groq", "llama-3.3-70b-versatile", settings)

    assert llm.model_name == "llama-3.3-70b-versatile"
    assert str(llm.openai_api_base) == settings.GROQ_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "gk-groq"


def test_get_llm_openrouter_uses_openrouter_base_url_and_key():
    settings = _make_settings()
    llm = get_llm("openrouter", "meta-llama/llama-3.3-70b-instruct", settings)

    assert llm.model_name == "meta-llama/llama-3.3-70b-instruct"
    assert str(llm.openai_api_base) == settings.OPENROUTER_BASE_URL
    assert llm.openai_api_key.get_secret_value() == "or-key"


def test_get_llm_unknown_provider_raises():
    settings = _make_settings()
    with pytest.raises(ValueError, match="Unknown provider: mystery"):
        get_llm("mystery", "some-model", settings)


def test_get_inference_llm_routes_to_groq_model():
    settings = _make_settings(PROVIDER="groq", GROQ_MODEL="qwen/qwen3-32b")
    llm = get_inference_llm(settings)
    assert llm.model_name == "qwen/qwen3-32b"


def test_get_inference_llm_routes_to_openrouter_model():
    settings = _make_settings(PROVIDER="openrouter", OPENROUTER_MODEL="qwen/qwen-2.5-72b-instruct")
    llm = get_inference_llm(settings)
    assert llm.model_name == "qwen/qwen-2.5-72b-instruct"


def test_get_judge_llm_uses_judge_model_override():
    settings = _make_settings(JUDGE_PROVIDER="groq", JUDGE_MODEL="llama-3.3-70b-versatile")
    llm = get_judge_llm(settings)
    assert llm.model_name == "llama-3.3-70b-versatile"


def test_get_judge_llm_falls_back_to_provider_model_when_judge_model_empty():
    settings = _make_settings(JUDGE_PROVIDER="openrouter", JUDGE_MODEL="", OPENROUTER_MODEL="fallback-model")
    llm = get_judge_llm(settings)
    assert llm.model_name == "fallback-model"


# get_node_llm — per-node overrides via env vars


def test_get_node_llm_without_env_falls_back_to_inference(monkeypatch):
    # Garante ausência de overrides de env
    for key in ("PLANNER_PROVIDER", "PLANNER_MODEL", "SQL_PROVIDER", "SQL_MODEL",
                "SYNTHESIS_PROVIDER", "SYNTHESIS_MODEL", "VERIFIER_PROVIDER", "VERIFIER_MODEL",
                "ROUTER_PROVIDER", "ROUTER_MODEL"):
        monkeypatch.delenv(key, raising=False)

    settings = _make_settings(PROVIDER="openrouter", OPENROUTER_MODEL="base-model")
    llm = get_node_llm("planner", settings)
    assert llm.model_name == "base-model"


def test_get_node_llm_with_env_override_uses_override(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "groq")
    monkeypatch.setenv("PLANNER_MODEL", "planner-specific-model")

    settings = _make_settings(PROVIDER="openrouter", OPENROUTER_MODEL="should-not-be-used")
    llm = get_node_llm("planner", settings)
    assert llm.model_name == "planner-specific-model"


def test_get_node_llm_override_only_affects_named_node(monkeypatch):
    monkeypatch.setenv("PLANNER_PROVIDER", "groq")
    monkeypatch.setenv("PLANNER_MODEL", "planner-model")
    monkeypatch.delenv("SQL_PROVIDER", raising=False)
    monkeypatch.delenv("SQL_MODEL", raising=False)

    settings = _make_settings(PROVIDER="openrouter", OPENROUTER_MODEL="default-model")

    planner_llm = get_node_llm("planner", settings)
    sql_llm = get_node_llm("sql", settings)

    assert planner_llm.model_name == "planner-model"
    assert sql_llm.model_name == "default-model"


def test_get_node_llm_provider_only_uses_provider_default_model(monkeypatch):
    # Se apenas PROVIDER for setado, usa o modelo default daquele provider
    monkeypatch.setenv("SYNTHESIS_PROVIDER", "groq")
    monkeypatch.delenv("SYNTHESIS_MODEL", raising=False)

    settings = _make_settings(
        PROVIDER="openai",
        OPENAI_MODEL="openai-default",
        GROQ_MODEL="groq-default",
    )
    llm = get_node_llm("synthesis", settings)
    assert llm.model_name == "groq-default"
