from __future__ import annotations

import json
from pathlib import Path

from voicebus.schema.events import VoiceBusEvent
from voicebus.schema.tasks import Task


def test_event_payload_examples_are_valid() -> None:
    payloads = json.loads(Path('examples/payloads/v3_events.json').read_text(encoding='utf-8'))
    for value in payloads.values():
        event = VoiceBusEvent(**value)
        assert event.schema_version == '3.0'


def test_task_schema_v3() -> None:
    task = Task(
        conversation_id='conv',
        turn_id='turn',
        task_signature='sig',
        text='hello',
        speaker_user='alice',
    )
    assert task.schema_version == '3.0'
    assert task.priority == 'interactive'
