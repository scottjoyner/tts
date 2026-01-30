from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ttsbench.models.base import BaseTTSModel, ModelCapabilities, SynthResult


class StubBase(BaseTTSModel):
    name = "stub"
    description = "Stub model"
    capabilities = ModelCapabilities(languages=["en"], supports_cloning=False, supports_styles=False)

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def availability_help(cls) -> str:
        return "This model is a stub. Follow documentation to add the implementation."

    def synth(self, text: str, config: Dict[str, Any], out_dir: Path) -> SynthResult:  # noqa: D401
        raise RuntimeError(f"Model {self.name} is a stub. {self.availability_help()}")


class StyleTTS2StubModel(StubBase):
    name = "styletts2"
    description = "StyleTTS2 (stub)"
    capabilities = ModelCapabilities(languages=["en"], supports_cloning=True, supports_styles=True)


class VITSStubModel(StubBase):
    name = "vits"
    description = "VITS baseline (stub)"
    capabilities = ModelCapabilities(languages=["en"], supports_cloning=True, supports_styles=False)


class BarkStubModel(StubBase):
    name = "bark"
    description = "Bark (stub)"
    capabilities = ModelCapabilities(languages=["en"], supports_cloning=False, supports_styles=True)


class QwenTTSStubModel(StubBase):
    name = "qwen_tts"
    description = "Qwen TTS (stub)"
    capabilities = ModelCapabilities(languages=["en", "zh"], supports_cloning=True, supports_styles=True)
