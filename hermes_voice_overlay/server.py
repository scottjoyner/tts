from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Hermes voice overlay requires fastapi and uvicorn. "
        "Install with: python -m pip install 'fastapi' 'uvicorn[standard]'"
    ) from exc


def _now_ms() -> int:
    return int(time.time() * 1000)


def _state_dir() -> Path:
    root = Path(os.environ.get("HERMES_HOME", Path.cwd() / ".hermes"))
    path = root / "voice_overlay"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_event(event: Dict[str, Any]) -> None:
    events_path = _state_dir() / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _extract_transcript(text: str) -> str:
    match = re.search(r"Transcript:\s*(.*?)(?:\. Workspace:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    transcript = match.group(1).strip() if match else text.strip()
    return re.sub(r"\s+", " ", transcript).strip(" .")


def _planner_base_url() -> str:
    return os.environ.get("HERMES_PLANNER_BASE_URL", "").strip().rstrip("/")


def _planner_model() -> str:
    return os.environ.get("HERMES_PLANNER_MODEL", "local-model").strip() or "local-model"


def _planner_timeout_seconds() -> float:
    raw = os.environ.get("HERMES_PLANNER_TIMEOUT_SECONDS", "20")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 20.0


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_plan(payload: Dict[str, Any], transcript: str) -> Dict[str, str]:
    prompt = str(payload.get("prompt") or transcript).strip()
    action = str(payload.get("action") or f"Queue Hermes agent work for: {prompt}").strip()
    spoken_response = str(
        payload.get("spoken_response") or f"I heard: {prompt}. I am asking Hermes to work on that now."
    ).strip()
    return {
        "prompt": prompt,
        "action": action,
        "spoken_response": spoken_response,
    }


def _fallback_prompt_plan(transcript: str) -> Dict[str, str]:
    prompt = transcript.strip()
    if not prompt:
        return {
            "prompt": "",
            "action": "No action queued because no transcript was provided.",
            "spoken_response": "I did not catch a command.",
        }
    return {
        "prompt": prompt,
        "action": f"Queue Hermes agent work for: {prompt}",
        "spoken_response": f"I heard: {prompt}. I am asking Hermes to work on that now.",
    }


def _lmstudio_prompt_plan(transcript: str) -> Dict[str, str]:
    base_url = _planner_base_url()
    if not base_url:
        return _fallback_prompt_plan(transcript)
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("HERMES_PLANNER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": _planner_model(),
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You turn live voice transcripts into production-safe Hermes agent work. "
                    "Return JSON only with string fields: prompt, action, spoken_response. "
                    "The prompt is the exact instruction Hermes should execute. The action is "
                    "a short status label. The spoken_response is one sentence telling the user "
                    "what will happen next."
                ),
            },
            {"role": "user", "content": f"Transcript: {transcript}"},
        ],
    }
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=_planner_timeout_seconds(),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _normalize_plan(_extract_json_object(str(content)), transcript)


def build_prompt_plan(transcript: str) -> Dict[str, str]:
    provider = os.environ.get("HERMES_PLANNER_PROVIDER", "fallback").strip().lower()
    if provider in {"lmstudio", "openai-compatible", "openai_compatible"}:
        try:
            plan = _lmstudio_prompt_plan(transcript)
            plan["planner_provider"] = "lmstudio"
            plan["planner_model"] = _planner_model()
            return plan
        except Exception as exc:
            plan = _fallback_prompt_plan(transcript)
            plan["planner_provider"] = "fallback"
            plan["planner_error"] = str(exc)
            return plan
    plan = _fallback_prompt_plan(transcript)
    plan["planner_provider"] = "fallback"
    return plan


def _request_plan(request: "VoiceExecuteRequest") -> Dict[str, str]:
    plan = request.metadata.get("plan") if isinstance(request.metadata, dict) else None
    if isinstance(plan, dict):
        return _normalize_plan(plan, request.transcript)
    return build_prompt_plan(request.prompt or request.transcript)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "hermes-voice-overlay"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: bool = False


class VoiceExecuteRequest(BaseModel):
    transcript: str
    prompt: str
    session_id: str = "voice-overlay"
    source: str = "sophia"
    dry_run: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Hermes Voice Overlay", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "hermes-voice-overlay",
        "planner_provider": os.environ.get("HERMES_PLANNER_PROVIDER", "fallback"),
        "planner_model": _planner_model(),
        "ts_ms": _now_ms(),
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
    user_text = _latest_user_message([message.model_dump() for message in request.messages])
    transcript = _extract_transcript(user_text)
    plan = build_prompt_plan(transcript)
    content = json.dumps(plan)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/voice-overlay/execute")
def execute(request: VoiceExecuteRequest) -> Dict[str, Any]:
    action_id = f"voice-{uuid.uuid4().hex[:12]}"
    plan = _request_plan(request)
    event = {
        "id": action_id,
        "ts_ms": _now_ms(),
        "type": "voice_overlay_action",
        "session_id": request.session_id,
        "source": request.source,
        "dry_run": request.dry_run,
        "transcript": request.transcript,
        "prompt": plan["prompt"],
        "action": plan["action"],
        "spoken_response": plan["spoken_response"],
        "metadata": request.metadata,
    }
    _append_event(event)
    return {
        "action_id": action_id,
        "status": "queued" if request.dry_run else "accepted",
        "dry_run": request.dry_run,
        "prompt": plan["prompt"],
        "action": plan["action"],
        "spoken_response": plan["spoken_response"],
        "planner_provider": plan.get("planner_provider", "metadata"),
        "planner_model": plan.get("planner_model"),
        "events_path": str(_state_dir() / "events.jsonl"),
    }


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(prog="hermes-voice-overlay")
    parser.add_argument("--host", default=os.environ.get("HERMES_VOICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HERMES_VOICE_PORT", "9720")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
