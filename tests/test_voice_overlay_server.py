from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

import hermes_voice_overlay.server as voice_server
from hermes_voice_overlay.server import app


def test_openai_compatible_prompt_planner() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "hermes-voice-overlay",
            "messages": [{"role": "user", "content": "Transcript: summarize the repo. Workspace: ."}],
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    payload = json.loads(content)
    assert payload["prompt"] == "summarize the repo"
    assert "Hermes" in payload["spoken_response"]


def test_lmstudio_prompt_planner(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "prompt": "run the unit tests for the voice overlay",
                                    "action": "Run voice overlay tests",
                                    "spoken_response": "I am running the voice overlay tests now.",
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(url: str, **kwargs: Any) -> Response:
        assert url == "http://lmstudio.test/v1/chat/completions"
        assert kwargs["json"]["model"] == "qwen-local"
        return Response()

    monkeypatch.setenv("HERMES_PLANNER_PROVIDER", "lmstudio")
    monkeypatch.setenv("HERMES_PLANNER_BASE_URL", "http://lmstudio.test")
    monkeypatch.setenv("HERMES_PLANNER_MODEL", "qwen-local")
    monkeypatch.setattr(voice_server.requests, "post", fake_post)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "hermes-voice-overlay",
            "messages": [{"role": "user", "content": "Transcript: test the voice overlay. Workspace: ."}],
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    payload = json.loads(content)
    assert payload["prompt"] == "run the unit tests for the voice overlay"
    assert payload["planner_provider"] == "lmstudio"


def test_voice_execute_records_action(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HERMES_PLANNER_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/voice-overlay/execute",
        json={
            "transcript": "run the tests",
            "prompt": "run the tests",
            "session_id": "demo",
            "source": "sophia",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    events = (tmp_path / "voice_overlay" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["prompt"] == "run the tests"


def test_voice_execute_uses_metadata_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/voice-overlay/execute",
        json={
            "transcript": "please inspect failures",
            "prompt": "please inspect failures",
            "session_id": "demo",
            "source": "sophia",
            "metadata": {
                "plan": {
                    "prompt": "inspect the failing CI job and summarize the fix",
                    "action": "Inspect failing CI",
                    "spoken_response": "I am checking the failing CI job now.",
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt"] == "inspect the failing CI job and summarize the fix"
    assert payload["spoken_response"] == "I am checking the failing CI job now."
