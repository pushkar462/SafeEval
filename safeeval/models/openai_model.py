"""OpenAI adapter (GPT-4o, GPT-3.5-turbo, etc.)."""
from typing import Optional
from .base import BaseLLM, ModelResponse

COST_MAP = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


class OpenAIModel(BaseLLM):
    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, **kwargs)
        try:
            from openai import OpenAI
            import os
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("pip install openai")

    def complete(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        response, latency = self._timed_call(_call)
        usage = response.usage
        in_cost, out_cost = COST_MAP.get(self.model_id, (0.005, 0.015))
        cost = ((usage.prompt_tokens * in_cost) + (usage.completion_tokens * out_cost)) / 1000

        return ModelResponse(
            text=response.choices[0].message.content or "",
            model_id=self.model_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency,
            cost_per_1k=cost * 1000,
            raw=response.model_dump(),
        )
