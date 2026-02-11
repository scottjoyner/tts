from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket

from tts_agent.agents.graph import AgentGraph
from tts_agent.api.routes_tasks import build_tasks_router
from tts_agent.config import Settings, settings
from tts_agent.events.ws_broadcast import WSConnectionHub
from tts_agent.ingest.checkpoint_store import IngestCheckpointStore
from tts_agent.ingest.jsonl_tailer import JSONLTailer
from tts_agent.ingest.stt_ws_client import STTWebSocketIngest
from tts_agent.memory.conversation_store import ConversationStore
from tts_agent.memory.summarizer import ConversationSummarizer
from tts_agent.storage.jsonl_writer import JSONLWriter
from tts_agent.tasks.cancel import is_cancel_intent, is_urgent_barge_in
from tts_agent.tasks.models import SegmentEvent, TaskCreate
from tts_agent.tasks.queue import TaskQueue
from tts_agent.tasks.router import SegmentRouter
from tts_agent.tasks.store_sqlite import TaskStore
from tts_agent.tts.pipeline import TTSPipeline
from tts_agent.tts.playback import InterruptiblePlayback
from tts_agent.tts.priority_queue import AudioPriorityQueue
from voicebus.schema.events import Actionability, VoiceBusEvent


class AppContext:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.store = TaskStore(cfg.task_db_path)
        self.queue = TaskQueue()
        self.router = SegmentRouter(cfg)
        self.memory = ConversationStore(cfg.task_db_path, max_turns=cfg.conversation_memory_turns)
        self.summarizer = ConversationSummarizer(self.memory, every_n_turns=cfg.conversation_summary_every)
        self.tts = TTSPipeline(cfg)
        self.event_log = JSONLWriter(cfg.events_jsonl_path)
        self.event_hub = WSConnectionHub()
        self.audio_hub = WSConnectionHub()
        self.worker_task: asyncio.Task | None = None
        self.ingest_tasks: list[asyncio.Task] = []
        self.checkpoints = IngestCheckpointStore(cfg.task_db_path)
        self.running: list[asyncio.Task] = []
        self.audio_queue = AudioPriorityQueue()
        self.playback = InterruptiblePlayback(self.audio_queue, self._broadcast_audio)
        self.playback_task: asyncio.Task | None = None
        self.graph = AgentGraph(self.store, Path('.'))

    async def _broadcast_audio(self, payload: dict, audio: bytes) -> None:
        await self.audio_hub.broadcast_json(payload)
        if audio:
            await self.audio_hub.broadcast_bytes(audio)

    def _conversation_id(self, event: SegmentEvent) -> str:
        if event.conversation_id:
            return event.conversation_id
        return f"{event.speaker.authenticated_user or event.speaker.candidate or 'anon'}-{event.timestamp_ms // 30000}"

    async def _emit(self, event: VoiceBusEvent) -> None:
        self.event_log.write(event.event_type, event.model_dump())
        await self.event_hub.broadcast_json(event.model_dump())

    async def process_event(self, payload: dict) -> None:
        if 'speaker' not in payload:
            payload['speaker'] = {
                'authenticated': bool(payload.get('authenticated', False)),
                'authenticated_user': payload.get('authenticated_user'),
                'candidate': payload.get('speaker_candidate'),
                'score': payload.get('speaker_score'),
            }
        if 'trigger_context' not in payload:
            payload['trigger_context'] = {'triggered': bool(payload.get('triggered', False))}
        if 'event_type' in payload and payload['event_type'] == 'segment_final' and 'turn_id' not in payload:
            payload['turn_id'] = payload.get('segment_id')
        event = SegmentEvent(**payload)
        event.ensure_actionability(self.router.is_actionable(event))
        await self._emit(event)

        if event.event_type not in {'turn_final', 'segment_final'}:
            return
        if not event.speaker.authenticated:
            return

        text = (event.transcript_final or event.text or '').strip()
        if is_cancel_intent(text):
            active = self.store.get_active_task()
            if active:
                self.store.cancel_task(active.task_id, reason='voice_cancel')
                self.playback.interrupt()
                await self._emit(VoiceBusEvent(event_type='task_cancelled', task_id=active.task_id, conversation_id=active.conversation_id))
                await self._emit(VoiceBusEvent(event_type='tts_interrupted', task_id=active.task_id, conversation_id=active.conversation_id))
                await self.audio_queue.put(active.task_id, 'Okay, I stopped.', priority='backchannel')
            return

        if not event.actionability.actionable:
            return
        if self.playback.active_task and is_urgent_barge_in(text):
            self.playback.interrupt()

        conversation_id = self._conversation_id(event)
        turn_id = event.turn_id or event.event_id
        self.memory.append_turn(conversation_id, 'user', text)
        self.summarizer.maybe_rollup(conversation_id)
        task = self.store.create_task(
            TaskCreate(
                source_segment_id=event.segment_id or turn_id,
                turn_id=turn_id,
                speaker_user=event.speaker.authenticated_user or event.speaker.candidate or self.cfg.tts_default_voice,
                text=text,
                conversation_id=conversation_id,
                priority='interactive',
            )
        )
        await self._emit(
            VoiceBusEvent(
                event_type='task_created',
                task_id=task.task_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                text=text,
                actionability=Actionability(actionable=True, confidence=event.actionability.confidence, reason=event.actionability.reason),
            )
        )
        await self.audio_queue.put(task.task_id, 'Okay—working on it.', priority='backchannel')
        await self.queue.put(task)

    async def _run_single_task(self, task):
        run_id = str(uuid.uuid4())
        await self._emit(VoiceBusEvent(event_type='task_started', task_id=task.task_id, run_id=run_id, conversation_id=task.conversation_id, turn_id=task.turn_id))
        result = await self.graph.run(task, run_id)
        if self.store.get_task(task.task_id).status != 'cancelled':
            self.store.update_task_status(task.task_id, 'done', result.summary)
            self.memory.append_turn(task.conversation_id or task.task_id, 'assistant', result.summary)
            await self._emit(VoiceBusEvent(event_type='agent_step_result', task_id=task.task_id, run_id=run_id, conversation_id=task.conversation_id, payload={'agent': 'summarizer', 'summary': result.summary}))
            await self.audio_queue.put(task.task_id, result.summary, priority='task_response')
            await self._emit(VoiceBusEvent(event_type='task_completed', task_id=task.task_id, run_id=run_id, conversation_id=task.conversation_id, text=result.summary))

    async def worker(self) -> None:
        while True:
            task = await self.queue.get()
            run_task = asyncio.create_task(self._run_single_task(task))
            self.running.append(run_task)


def create_app(cfg: Settings = settings) -> FastAPI:
    app_context = AppContext(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_context.worker_task = asyncio.create_task(app_context.worker())
        app_context.playback_task = asyncio.create_task(app_context.playback.run(app_context.tts.synthesize_stream))
        if cfg.stt_ingest_mode in {'ws', 'both'}:
            ws_ingest = STTWebSocketIngest(cfg.stt_ws_url, app_context.process_event, checkpoints=app_context.checkpoints)
            app_context.ingest_tasks.append(asyncio.create_task(ws_ingest.run()))
        if cfg.stt_ingest_mode in {'tail', 'both'}:
            tailer = JSONLTailer(cfg.stt_jsonl_dir, app_context.process_event, checkpoints=app_context.checkpoints)
            app_context.ingest_tasks.append(asyncio.create_task(tailer.run()))
        yield
        for task in app_context.ingest_tasks + app_context.running:
            task.cancel()
        if app_context.worker_task:
            app_context.worker_task.cancel()
        if app_context.playback_task:
            app_context.playback_task.cancel()

    app = FastAPI(title='TTS Agent Orchestrator', lifespan=lifespan)
    app.include_router(build_tasks_router(app_context.store, app_context.queue))

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/stats')
    async def stats() -> dict:
        data = app_context.store.stats()
        data['queue_depth'] = app_context.queue.qsize()
        data['audio_queue_depth'] = app_context.audio_queue.qsize()
        return data

    @app.get('/runs/{run_id}/trace')
    async def run_trace(run_id: str) -> dict:
        return {'run_id': run_id, 'note': 'use /trace/{task_id} for full detail'}

    @app.get('/trace/{task_id}')
    async def trace(task_id: str) -> dict:
        task = app_context.store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail='Task not found')
        return app_context.store.task_trace(task_id)

    @app.get('/tasks')
    async def list_tasks(status: str | None = None) -> list[dict]:
        return [task.model_dump() for task in app_context.store.list_tasks(status=status)]

    @app.get('/tasks/{task_id}')
    async def get_task(task_id: str) -> dict:
        task = app_context.store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail='Task not found')
        return task.model_dump()

    @app.post('/tasks/enqueue')
    async def enqueue_task(payload: dict) -> dict:
        task = app_context.store.create_task(
            TaskCreate(
                source_segment_id=payload.get('source_segment_id', 'manual'),
                speaker_user=payload.get('speaker_user', cfg.tts_default_voice),
                text=payload['text'],
                conversation_id=payload.get('conversation_id', 'manual-conversation'),
                turn_id=payload.get('turn_id', payload.get('source_segment_id', 'manual-turn')),
                priority=payload.get('priority', 'interactive'),
            )
        )
        await app_context.queue.put(task)
        return task.model_dump()

    @app.post('/tasks/{task_id}/cancel')
    async def cancel_task(task_id: str) -> dict:
        if not app_context.store.cancel_task(task_id):
            raise HTTPException(status_code=404, detail='Task not cancellable')
        app_context.playback.interrupt()
        return {'task_id': task_id, 'status': 'cancelled'}

    @app.websocket('/ws/events')
    async def ws_events(websocket: WebSocket) -> None:
        await app_context.event_hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            app_context.event_hub.disconnect(websocket)

    @app.get('/events/recent')
    async def events_recent(limit: int = 20) -> list[dict]:
        if not cfg.events_jsonl_path.exists():
            return []
        lines = cfg.events_jsonl_path.read_text(encoding='utf-8').splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    @app.websocket('/ws/audio')
    async def ws_audio(websocket: WebSocket) -> None:
        await app_context.audio_hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            app_context.audio_hub.disconnect(websocket)

    return app
