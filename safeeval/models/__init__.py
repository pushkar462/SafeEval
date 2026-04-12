"""Model factory — load from config."""
from typing import Optional
import yaml
from .base import BaseLLM, ModelResponse


def load_model(model_name: str, config_path: str = "config/models.yaml", api_key: Optional[str] = None) -> BaseLLM:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["models"].get(model_name)
    if not model_cfg:
        raise ValueError(f"Model '{model_name}' not found in {config_path}")

    provider = model_cfg["provider"]
    kwargs = dict(
        model_id=model_cfg["model_id"],
        max_tokens=model_cfg.get("max_tokens", 1024),
        temperature=model_cfg.get("temperature", 0.0),
    )

    if provider == "openai":
        from .openai_model import OpenAIModel
        return OpenAIModel(api_key=api_key, **kwargs)
    elif provider == "anthropic":
        from .anthropic_model import AnthropicModel
        return AnthropicModel(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["BaseLLM", "ModelResponse", "load_model"]
