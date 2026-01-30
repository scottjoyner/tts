from __future__ import annotations

from typing import Dict, Iterable, Type

from ttsbench.models.base import BaseTTSModel
from ttsbench.models.plugins.coqui_xtts import CoquiXTTSModel
from ttsbench.models.plugins.piper import PiperModel
from ttsbench.models.plugins.stub import (
    BarkStubModel,
    QwenTTSStubModel,
    StyleTTS2StubModel,
    VITSStubModel,
)

MODEL_REGISTRY: Dict[str, Type[BaseTTSModel]] = {
    CoquiXTTSModel.name: CoquiXTTSModel,
    PiperModel.name: PiperModel,
    StyleTTS2StubModel.name: StyleTTS2StubModel,
    VITSStubModel.name: VITSStubModel,
    BarkStubModel.name: BarkStubModel,
    QwenTTSStubModel.name: QwenTTSStubModel,
}


def list_models() -> Iterable[Type[BaseTTSModel]]:
    return MODEL_REGISTRY.values()


def get_model(name: str) -> Type[BaseTTSModel]:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]
