FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tts_agent ./tts_agent
COPY ttsbench ./ttsbench
COPY voicebus ./voicebus

RUN pip install --no-cache-dir .

RUN groupadd --gid 10001 ttsagent && \
    useradd --create-home --uid 10001 --gid 10001 ttsagent && \
    mkdir -p /data /models /data/logs /data/artifacts && \
    chown -R ttsagent:ttsagent /app /data /models

USER ttsagent

EXPOSE 8010

VOLUME ["/data", "/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3).read()"

CMD ["tts-agent", "run"]
