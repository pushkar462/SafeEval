"""Anthropic adapter (Claude models)."""
from typing import Optional
from .base import BaseLLM, ModelResponse

COST_MAP = {
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


class AnthropicModel(BaseLLM):
    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, **kwargs)
        try:
            import anthropic as sdk
            import os
            self.client = sdk.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("pip install anthropic")

    def complete(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        kwargs = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        def _call():
            return self.client.messages.create(**kwargs)

        response, latency = self._timed_call(_call)
        usage = response.usage
        in_cost, out_cost = COST_MAP.get(self.model_id, (0.003, 0.015))
        cost = ((usage.input_tokens * in_cost) + (usage.output_tokens * out_cost)) / 1000

        return ModelResponse(
            text=response.content[0].text if response.content else "",
            model_id=self.model_id,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency,
            cost_per_1k=cost * 1000,
            raw={},
        )
