from dataclasses import dataclass

@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

def normalize_provider(value: str) -> str:
    value = value.strip().lower()
    value = {"anthorpic": "anthropic", "google": "gemini", "openai-compatible": "custom"}.get(value, value)
    if value not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return value

def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if provider in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI
        if provider == "custom" and not config.base_url:
            raise ValueError("custom requires CUSTOM_BASE_URL")
        if provider == "openrouter":
            kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**kwargs)
    from langchain_ollama import ChatOllama
    kwargs.pop("api_key", None)
    return ChatOllama(**kwargs)
