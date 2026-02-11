from __future__ import annotations

from dataclasses import dataclass

from tts_agent.llm.openai_compat_client import OpenAICompatClient


@dataclass
class ProviderModel:
    name: str
    base_url: str


class LLMProviderRegistry:
    def create_client(self, model: ProviderModel) -> OpenAICompatClient:
        return OpenAICompatClient(base_url=model.base_url)
