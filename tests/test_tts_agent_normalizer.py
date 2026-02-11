from tts_agent.ingest.normalizer import normalize_event


def test_normalize_segment_final_defaults() -> None:
    payload = {
        'type': 'segment_final',
        'segment_id': 'seg-1',
        'transcript_final': 'Can you summarize this?',
        'authenticated': True,
        'speaker_score': 0.91,
    }
    event = normalize_event(payload)
    assert event.event_type == 'segment_final'
    assert event.is_command is False
    assert event.authenticated is True
    assert event.segment_id == 'seg-1'
