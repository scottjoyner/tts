from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class SynthResult:
    audio_path: Path
    sample_rate: int
    timings: Dict[str, float]
    stats: Dict[str, float]


@dataclass(frozen=True)
class ModelCapabilities:
    languages: Iterable[str]
    supports_cloning: bool
    supports_styles: bool


class BaseTTSModel(abc.ABC):
    name: str
    description: str
    capabilities: ModelCapabilities

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def availability_help(cls) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def synth(self, text: str, config: Dict[str, Any], out_dir: Path) -> SynthResult:
        raise NotImplementedError
