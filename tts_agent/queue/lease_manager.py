from __future__ import annotations

import time


class LeaseManager:
    def __init__(self, lease_ms: int = 15000) -> None:
        self.lease_ms = lease_ms

    def new_expiry(self) -> int:
        return int(time.time() * 1000) + self.lease_ms

    def heartbeat(self) -> int:
        return int(time.time() * 1000)

    def is_expired(self, expires_at_ms: int) -> bool:
        return int(time.time() * 1000) > expires_at_ms
