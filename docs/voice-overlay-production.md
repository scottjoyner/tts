# Hermes Voice Overlay Production Container

Hermes exposes a voice-overlay gateway for Sophia. The gateway has two roles:

- OpenAI-compatible planning at `/v1/chat/completions`.
- Execution-intent capture at `/voice-overlay/execute`.

The gateway is intentionally narrow. It accepts a final voice transcript,
returns a concise prompt/action plan, records the action intent, and returns a
spoken status response for Sophia.

## LM Studio Planning

The gateway can use an LM Studio local server to decide the exact Hermes prompt
and the spoken follow-up text. Configure it with:

```env
HERMES_PLANNER_PROVIDER=lmstudio
HERMES_PLANNER_BASE_URL=http://host.docker.internal:1234
HERMES_PLANNER_MODEL=local-model
HERMES_PLANNER_TIMEOUT_SECONDS=20
```

The LM Studio endpoint must implement the OpenAI-compatible
`/v1/chat/completions` API. Hermes asks the model for JSON with:

```json
{
  "prompt": "exact instruction for Hermes",
  "action": "short action label",
  "spoken_response": "one sentence Sophia should speak"
}
```

If LM Studio is unavailable or returns invalid JSON, the gateway falls back to
a deterministic transcript-to-prompt plan so the voice path still produces an
auditable event.

## Container Command

The standard Hermes image can run the overlay directly:

```bash
docker run --rm \
  -p 127.0.0.1:9720:9720 \
  --add-host host.docker.internal:host-gateway \
  -e HERMES_PLANNER_PROVIDER=lmstudio \
  -e HERMES_PLANNER_BASE_URL=http://host.docker.internal:1234 \
  -e HERMES_PLANNER_MODEL=local-model \
  -v hermes-data:/opt/data \
  hermes-agent:voice-overlay-prod \
  hermes-voice-overlay --host 0.0.0.0 --port 9720
```

## Compose

Sophia owns the full production stack at:

```text
Sophia/deploy/prod/docker-compose.yml
```

That stack builds this repository as `hermes-agent:voice-overlay-prod` and runs:

```bash
hermes-voice-overlay --host 0.0.0.0 --port 9720
```

## Setup Script Onboarding

`setup-hermes.sh` now offers voice overlay onboarding on desktop/server hosts.
The optional flow writes the voice-related `.env` values, checks Docker Compose,
and can build/start this container:

```bash
./setup-hermes.sh
```

For unattended setup:

```bash
HERMES_SETUP_VOICE=1 HERMES_START_VOICE=1 ./setup-hermes.sh
```

This only starts the Hermes voice gateway. The full microphone-to-agent loop
also needs Sophia's production stack, because Sophia owns browser microphone
capture, VAD/STT, and spoken responses.

## Health

```bash
curl http://localhost:9720/health
```

Expected response:

```json
{"ok": true, "service": "hermes-voice-overlay", "planner_provider": "lmstudio"}
```

## Data

The overlay writes JSONL action records to:

```text
$HERMES_HOME/voice_overlay/events.jsonl
```

In the container this resolves to:

```text
/opt/data/voice_overlay/events.jsonl
```

## Safety Mode

Sophia currently calls `/voice-overlay/execute` with `dry_run: true`. The
gateway records the action but does not directly execute arbitrary agent work.
Keep this default until the voice UX includes confirmation, allow-listing, and
auditable cancellation.
