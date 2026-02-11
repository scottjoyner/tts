# TTSBench + TTS Agent Orchestrator (v2)

This repository contains:

1. `ttsbench`: benchmarking/training utilities.
2. `tts_agent`: local-first realtime STT→task→RALPH→TTS orchestrator with replay-safe ingest, bounded conversation memory, and interruptible speech.

## v2 Orchestrator Tree (new/updated)

```text
tts_agent/
  api/
    routes_tasks.py                # cancel endpoints
  ingest/
    checkpoint_store.py            # event_id + hash dedupe + offset checkpoints
    replay_client.py               # replay cursor + JSONL offset reader
    stt_ws_client.py               # reconnect/resubscribe + replay cursor + dedupe
    jsonl_tailer.py                # offset checkpoint replay from disk
  memory/
    conversation_store.py          # raw transcript + summary tables
    summarizer.py                  # rolling summary every N turns
  ralph/
    context_builder.py             # bounded context construction
    iteration.py                   # IterationResult schema
    completion.py                  # done-when rule checks
    scoring.py                     # judge/heuristic confidence score
    executor.py                    # structured loop + safety valves + fallback
  llm/
    router.py                      # per-stage routing + health + fallback chain
  tasks/
    state_machine.py               # lifecycle transitions
    cancel.py                      # cancel/barge-in intents + active registry
    models.py                      # extended task + segment fields
    store_sqlite.py                # persisted lifecycle/trace/artifacts helpers
  tts/
    priority_queue.py              # system_critical > task_response > backchannel
    playback.py                    # interruptible foreground playback
    pipeline.py                    # chunk/stream + voice profile plumbing
    engines/
      qwen3_tts_engine.py          # voice_profile + synthesize_stream contract
```

## Run against STT upstream

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run orchestrator:

```bash
tts-agent run
```

Example env:

```bash
export STT_WS_URL=ws://127.0.0.1:8000/ws/events
export STT_INGEST_MODE=ws              # ws|tail|both
export TASK_DB_PATH=data/tasks.db
export TTS_ENGINE=dummy                # qwen3|dummy
```

## Replay + dedupe behavior

- `ingest_checkpoints` table stores per-source `last_event_id` and JSONL byte offsets.
- `ingest_dedupe` table stores SHA-256 event hash; duplicate payloads are skipped.
- WebSocket ingest re-subscribes with `?cursor=<last_event_id>` when available.
- JSONL ingest resumes from last persisted byte offset.

## Voice profile enrollment/mapping

- Set speaker identity in upstream auth (`authenticated_user`) or pass `speaker_user` on manual `POST /tasks`.
- The TTS pipeline forwards `voice_profile` through engine APIs.
- `Qwen3TTSEngine` exposes:
  - `synthesize(text, sample_rate, voice_profile)`
  - `synthesize_stream(text, sample_rate, voice_profile)`

## Cancellation + barge-in test flow

1. Start a long task (voice or `POST /tasks`).
2. Speak authenticated cancel intent: “stop”, “cancel”, “nevermind”.
3. Active task transitions to `cancelled`; playback interrupts immediately.
4. During playback, urgent authenticated phrase (“stop”, “wait”, “new request”) triggers barge-in interrupt.

## Main APIs

- `POST /tasks/{id}/cancel`
- `POST /tasks/cancel-active`
- `GET /trace/{task_id}`
- `GET /stats` (queue depth, avg duration, model health, TTS TTFB)
- `WS /ws/events` (task lifecycle, iterations, routing decisions)
- `WS /ws/audio` (audio chunk + interruption events)

## Event payload examples

### Task creation from authenticated transcript

```json
{
  "event_type": "task_created",
  "task": {
    "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "source_segment_id": "seg-192",
    "speaker_user": "primary_user",
    "conversation_id": "primary_user-44",
    "status": "queued",
    "text": "Summarize my latest meeting notes"
  }
}
```

### RALPH iteration event

```json
{
  "event_type": "ralph_iteration",
  "iteration": 2,
  "plan": ["gather context", "draft summary", "verify done"],
  "next_action": "draft summary",
  "completion_check": {"done": false, "reason": "needs_more_work"},
  "confidence": 0.67
}
```

### Audio chunk event

```json
{
  "event_type": "tts_chunk",
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "priority": "task_response",
  "text": "Here is the summary so far.",
  "bytes": 8192
}
```
