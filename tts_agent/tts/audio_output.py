from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LocalAudioOutput:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def play(self, pcm_bytes: bytes) -> None:
        if not self.enabled:
            return
        logger.info('Local playback enabled; bytes=%s', len(pcm_bytes))
