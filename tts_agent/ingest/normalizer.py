from __future__ import annotations

from typing import Any

from tts_agent.tasks.models import SegmentEvent


def normalize_event(payload: dict[str, Any]) -> SegmentEvent:
    event_type = payload.get('event_type') or payload.get('type') or 'unknown'
    return SegmentEvent(
        event_type=event_type,
        segment_id=payload.get('segment_id'),
        transcript_final=payload.get('transcript_final'),
        transcript_partial=payload.get('transcript_partial'),
        triggered=bool(payload.get('triggered', False)),
        is_command=bool(payload.get('is_command', payload.get('triggered', False))),
        authenticated=bool(payload.get('authenticated', False)),
        authenticated_user=payload.get('authenticated_user'),
        speaker_candidate=payload.get('speaker_candidate'),
        speaker_score=payload.get('speaker_score'),
        start_ts=payload.get('start_ts'),
        end_ts=payload.get('end_ts'),
        raw=payload,
    )
