from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def retry_with_backoff(
    fn: Callable[[], Awaitable[None]],
    retries: int = 10,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> None:
    delay = base_delay
    for attempt in range(retries):
        try:
            await fn()
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
