import os
from dataclasses import dataclass
from pathlib import Path
from model_provider import ProviderConfig, normalize_provider

@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig

def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env")
    def provider_config(prefix, fallback=None):
        provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
        defaults = {"openai": "gpt-4o-mini", "custom": "gpt-4o-mini", "gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-5", "ollama": "llama3.2", "openrouter": "openai/gpt-4o-mini"}
        model = os.getenv(f"{prefix}_MODEL", fallback.model_name if fallback and provider == fallback.provider else defaults[provider])
        return ProviderConfig(provider, model, float(os.getenv(f"{prefix}_TEMPERATURE", "0")), os.getenv(f"{provider.upper()}_API_KEY"), os.getenv(f"{provider.upper()}_BASE_URL"))
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200"))
    keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))
    if threshold <= 0 or keep < 1:
        raise ValueError("Compact threshold must be positive and keep_messages >= 1")
    model = provider_config("LLM")
    return LabConfig(root, root / "data", state, threshold, keep, model, provider_config("JUDGE", model))
