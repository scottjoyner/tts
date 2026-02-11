# TTSBench + TTS Agent Orchestrator

This repository now contains two local-first components:

1. `ttsbench`: TTS benchmarking/training utilities.
2. `tts_agent`: a realtime **Agent + TTS Orchestrator** that consumes STT events, gates on speaker authentication, runs a RALPH-style loop, and synthesizes speech responses.

## Repo Tree (Orchestrator)

```text
tts_agent/
  main.py
  server.py
  config.py
  ingest/
    stt_ws_client.py
    jsonl_tailer.py
    normalizer.py
  tasks/
    models.py
    store_sqlite.py
    queue.py
    router.py
  ralph/
    executor.py
    planner.py
    evaluator.py
    router.py
    prompts.py
  llm/
    openai_compat_client.py
    providers.py
  tts/
    pipeline.py
    chunker.py
    audio_output.py
    engines/
      qwen3_tts_engine.py
      dummy_engine.py
    voice_profiles.py
  events/
    bus.py
    ws_broadcast.py
  storage/
    jsonl_writer.py
    artifact_store.py
  utils/
    logging.py
    time.py
    retry.py
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Run API + orchestrator worker:

```bash
tts-agent run
```

Connect to STT websocket by setting:

```bash
export STT_WS_URL=ws://127.0.0.1:8000/ws/events
export STT_INGEST_MODE=ws
```

Or use JSONL tailing mode:

```bash
export STT_INGEST_MODE=tail
export STT_JSONL_DIR=/path/to/stt/data
```

## Behavior Summary

- Observes all `segment_final` events and logs them to JSONL.
- Only authenticated + actionable requests enter task execution.
- Authenticated requests become persistent tasks in SQLite (`tasks`, `task_events`, `llm_calls`, `tts_outputs`).
- RALPH executor loops until completion/limits.
- TTS pipeline chunks text and streams audio on `/ws/audio`.

## API Endpoints

- `GET /health`
- `GET /stats`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks`
- `WS /ws/events`
- `WS /ws/audio`

## qwen3-tts Integration Contract

The wrapper is `tts_agent/tts/engines/qwen3_tts_engine.py`.

Expected contract for your benchmark repo adapter:

- Input: `(text: str, sample_rate: int, voice_profile_path: Optional[path])`
- Output: `bytes` containing streaming PCM16 audio chunks (or single chunk for v1).
- Non-blocking API: `async def synthesize(...) -> bytes`.

Where to place model files:

- Put checkpoints under your local model path, e.g. `models/qwen3-tts/`.
- Pass path through env/config and initialize `Qwen3TTSEngine(model_path=...)`.

Validation command (manual):

```bash
python - <<'PY'
import asyncio
from tts_agent.tts.engines.qwen3_tts_engine import Qwen3TTSEngine

async def main():
    engine = Qwen3TTSEngine()
    audio = await engine.synthesize('Hello from qwen3-tts adapter', sample_rate=24000)
    print('bytes=', len(audio))

asyncio.run(main())
PY
```

## Troubleshooting Latency

- Lower `TTS_CHUNK_SENTENCES` to `1` for faster first audio.
- Use lower-latency planning model in `routing.json`.
- Disable local playback if headless (`TTS_PLAY_LOCAL=0`).

## Existing TTSBench Commands

`ttsbench` commands are still available for benchmarking/training workflows.
