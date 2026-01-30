from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.sql import insert


@dataclass
class RunInfo:
    run_id: str
    created_at: datetime
    prompts_path: str
    seed: int
    notes: str | None = None


class ResultsWriter:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.engine = create_engine(f"sqlite:///{sqlite_path}")
        self.metadata = MetaData()
        self._init_tables()
        self.metadata.create_all(self.engine)

    def _init_tables(self) -> None:
        self.runs = Table(
            "runs",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("created_at", DateTime),
            Column("prompts_path", String),
            Column("seed", Integer),
            Column("notes", String),
        )
        self.models = Table(
            "models",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, ForeignKey("runs.id")),
            Column("name", String),
            Column("description", String),
            Column("available", Boolean),
        )
        self.prompts = Table(
            "prompts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, ForeignKey("runs.id")),
            Column("prompt_id", String),
            Column("text", String),
            Column("language", String),
            Column("style", String),
        )
        self.outputs = Table(
            "outputs",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, ForeignKey("runs.id")),
            Column("model_id", Integer, ForeignKey("models.id")),
            Column("prompt_id", Integer, ForeignKey("prompts.id")),
            Column("audio_path", String),
            Column("sample_rate", Integer),
        )
        self.metrics = Table(
            "metrics",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("output_id", Integer, ForeignKey("outputs.id")),
            Column("name", String),
            Column("value", Float),
        )

    def write_run(self, run: RunInfo) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(self.runs).values(
                    id=run.run_id,
                    created_at=run.created_at,
                    prompts_path=run.prompts_path,
                    seed=run.seed,
                    notes=run.notes,
                )
            )

    def write_models(self, run_id: str, models: Iterable[Dict[str, object]]) -> List[int]:
        ids: List[int] = []
        with self.engine.begin() as conn:
            for model in models:
                result = conn.execute(
                    insert(self.models).values(
                        run_id=run_id,
                        name=model["name"],
                        description=model["description"],
                        available=bool(model["available"]),
                    )
                )
                ids.append(int(result.inserted_primary_key[0]))
        return ids

    def write_prompts(self, run_id: str, prompts: Iterable[Dict[str, object]]) -> List[int]:
        ids: List[int] = []
        with self.engine.begin() as conn:
            for prompt in prompts:
                result = conn.execute(
                    insert(self.prompts).values(
                        run_id=run_id,
                        prompt_id=prompt["id"],
                        text=prompt["text"],
                        language=prompt["language"],
                        style=prompt["style"],
                    )
                )
                ids.append(int(result.inserted_primary_key[0]))
        return ids

    def write_output(
        self,
        run_id: str,
        model_id: int,
        prompt_id: int,
        audio_path: str,
        sample_rate: int,
        metrics: Dict[str, float],
    ) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self.outputs).values(
                    run_id=run_id,
                    model_id=model_id,
                    prompt_id=prompt_id,
                    audio_path=audio_path,
                    sample_rate=sample_rate,
                )
            )
            output_id = int(result.inserted_primary_key[0])
            for name, value in metrics.items():
                conn.execute(
                    insert(self.metrics).values(output_id=output_id, name=name, value=value)
                )

    def dump_json(self, path: Path, payload: Dict[str, object]) -> None:
        path.write_text(json.dumps(payload, indent=2))
