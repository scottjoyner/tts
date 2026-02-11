from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import websockets


aSYNC_BACKCHANNEL = 'Okay—working on it.'


async def run(stt_ws: str, agent_api: str) -> None:
    async with websockets.connect(stt_ws) as ws:
        print(f'Connected to {stt_ws}')
        async for raw in ws:
            event = json.loads(raw)
            if event.get('event_type') not in {'turn_final', 'segment_final'}:
                continue
            actionable = bool(event.get('actionability', {}).get('actionable'))
            text = event.get('transcript_final') or event.get('text')
            if not actionable or not text:
                continue
            payload = {
                'conversation_id': event.get('conversation_id', 'demo-conv'),
                'turn_id': event.get('turn_id', event.get('event_id', 'demo-turn')),
                'source_segment_id': event.get('segment_id', event.get('turn_id', 'demo-segment')),
                'speaker_user': (event.get('speaker') or {}).get('authenticated_user', 'demo-user'),
                'text': text,
                'priority': 'interactive',
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(f'{agent_api}/tasks/enqueue', json=payload)
                resp.raise_for_status()
            print('task_created', resp.json().get('task_id'))
            print('tts_backchannel', aSYNC_BACKCHANNEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stt-ws', default='ws://localhost:8765/ws/events')
    parser.add_argument('--agent-api', default='http://localhost:8000')
    args = parser.parse_args()
    asyncio.run(run(args.stt_ws, args.agent_api))
