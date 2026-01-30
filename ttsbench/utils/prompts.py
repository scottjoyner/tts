from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class PromptItem(BaseModel):
    id: str
    text: str
    language: str = "en"
    style: Optional[str] = None


class PromptConfig(BaseModel):
    sample_rate: int = 24000
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_duration_s: Optional[float] = None
    speaker: Optional[str] = None
    voice: Optional[str] = None
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    speaker_wav: Optional[str] = None
    styles: List[str] = Field(default_factory=list)
    prompts: List[PromptItem]


class PromptSet(BaseModel):
    config: PromptConfig


DEFAULT_STYLES = ["neutral", "excited", "whisper", "fast", "slow"]


def load_prompts(path: Path) -> PromptSet:
    data = yaml.safe_load(path.read_text())
    if "styles" not in data:
        data["styles"] = DEFAULT_STYLES
    prompt_config = PromptConfig(
        sample_rate=data.get("sample_rate", 24000),
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        max_duration_s=data.get("max_duration_s"),
        speaker=data.get("speaker"),
        voice=data.get("voice"),
        model_name=data.get("model_name"),
        model_path=data.get("model_path"),
        speaker_wav=data.get("speaker_wav"),
        styles=data.get("styles", DEFAULT_STYLES),
        prompts=[PromptItem(**item) for item in data["prompts"]],
    )
    return PromptSet(config=prompt_config)


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def prompt_metadata(prompt: PromptItem) -> Dict[str, str]:
    return {
        "id": prompt.id,
        "language": prompt.language,
        "style": prompt.style or "neutral",
    }
