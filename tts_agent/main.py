from __future__ import annotations

import typer
import uvicorn

from tts_agent.config import settings
from tts_agent.server import create_app
from tts_agent.utils.logging import configure_logging

app = typer.Typer(help='TTS Agent Orchestrator CLI')


@app.command('run')
def run() -> None:
    configure_logging()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == '__main__':
    app()
