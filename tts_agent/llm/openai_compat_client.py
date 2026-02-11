from __future__ import annotations

import httpx


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str = 'dummy') -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    async def chat(self, model: str, prompt: str, timeout: float = 30.0) -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f'{self.base_url}/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}'},
                json={
                    'model': model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload['choices'][0]['message']['content']
