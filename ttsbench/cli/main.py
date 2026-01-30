from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from ttsbench.metrics.audio_metrics import AudioMetrics
from ttsbench.metrics.asr_metrics import compute_asr_metrics
from ttsbench.metrics.speaker_similarity import cosine_similarity
from ttsbench.models.base import BaseTTSModel
from ttsbench.models.registry import get_model, list_models
from ttsbench.training.prep import prepare_dataset
from ttsbench.training.recipes import create_training_plan
from ttsbench.utils.logging import setup_logging
from ttsbench.utils.prompts import load_prompts, normalize_prompt
from ttsbench.utils.report import write_report
from ttsbench.utils.results import ResultsWriter, RunInfo

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _root(
    log_path: Optional[Path] = typer.Option(None, help="Optional JSON log file."),
) -> None:
    setup_logging(log_path)


@app.command("list-models")
def list_models_cmd() -> None:
    table = Table(title="TTS Models")
    table.add_column("Name")
    table.add_column("Available")
    table.add_column("Description")
    for model_cls in list_models():
        available = "yes" if model_cls.is_available() else "no"
        table.add_row(model_cls.name, available, model_cls.description)
    console.print(table)


@app.command("download")
def download_cmd(model: str) -> None:
    model_cls = get_model(model)
    if model_cls.is_available():
        console.print(f"{model} is already available locally.")
        return
    console.print(model_cls.availability_help())


def _run_id(run_id: Optional[str]) -> str:
    return run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@app.command("synth")
def synth_cmd(
    model: str,
    prompts: Path,
    out: Path,
    run_id: Optional[str] = typer.Option(None, help="Explicit run id."),
    seed: int = typer.Option(1337, help="Random seed."),
) -> None:
    run_id = _run_id(run_id)
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    prompt_set = load_prompts(prompts)
    model_cls = get_model(model)
    if not model_cls.is_available():
        raise typer.Exit(model_cls.availability_help())

    model_instance = model_cls()
    config = prompt_set.config.model_dump()

    for prompt in prompt_set.config.prompts:
        text = normalize_prompt(prompt.text)
        styles = [prompt.style] if prompt.style else prompt_set.config.styles
        for style in styles:
            synth_dir = run_dir / model / prompt.id / style
            output_file = synth_dir / "audio.wav"
            if output_file.exists():
                logger.info("Skipping existing output", extra={"path": str(output_file)})
                continue
            config["style"] = style
            result = model_instance.synth(text=text, config=config, out_dir=synth_dir)
            logger.info(
                "Synth complete",
                extra={"path": str(result.audio_path), "timings": result.timings},
            )


def _benchmark_run(
    models: List[str],
    prompts: Path,
    out: Path,
    run_id: str,
    seed: int,
    reference_voice: Optional[Path],
    config_override: Optional[Dict[str, object]] = None,
) -> None:
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_set = load_prompts(prompts)
    random.seed(seed)

    results_writer = ResultsWriter(run_dir / "results.sqlite")
    run_info = RunInfo(
        run_id=run_id,
        created_at=datetime.utcnow(),
        prompts_path=str(prompts),
        seed=seed,
    )
    results_writer.write_run(run_info)

    model_rows = []
    model_instances: Dict[str, BaseTTSModel] = {}
    for name in models:
        model_cls = get_model(name)
        available = model_cls.is_available()
        model_rows.append(
            {
                "name": model_cls.name,
                "description": model_cls.description,
                "available": available,
            }
        )
        if available:
            model_instances[name] = model_cls()
    model_ids = results_writer.write_models(run_id, model_rows)
    model_id_lookup = {row["name"]: model_ids[idx] for idx, row in enumerate(model_rows)}

    prompt_rows = []
    for prompt in prompt_set.config.prompts:
        prompt_rows.append(
            {
                "id": prompt.id,
                "text": normalize_prompt(prompt.text),
                "language": prompt.language,
                "style": prompt.style or "neutral",
            }
        )
    prompt_ids = results_writer.write_prompts(run_id, prompt_rows)
    prompt_id_lookup = {row["id"]: prompt_ids[idx] for idx, row in enumerate(prompt_rows)}

    outputs_payload: List[Dict[str, object]] = []
    config = prompt_set.config.model_dump()
    if config_override:
        config.update(config_override)
    for name, model_instance in model_instances.items():
        for prompt in prompt_set.config.prompts:
            text = normalize_prompt(prompt.text)
            styles = [prompt.style] if prompt.style else prompt_set.config.styles
            for style in styles:
                synth_dir = run_dir / name / prompt.id / style
                output_path = synth_dir / "audio.wav"
                if output_path.exists():
                    logger.info("Skipping existing output", extra={"path": str(output_path)})
                    audio_path = output_path
                    sample_rate = prompt_set.config.sample_rate
                    timings = {}
                else:
                    config["style"] = style
                    config["language"] = prompt.language
                    result = model_instance.synth(text=text, config=config, out_dir=synth_dir)
                    audio_path = result.audio_path
                    sample_rate = result.sample_rate
                    timings = result.timings

                metrics = AudioMetrics(audio_path).compute()
                metrics.update(timings)
                asr_metrics = compute_asr_metrics(audio_path, text, prompt.language)
                if asr_metrics:
                    metrics.update(asr_metrics)
                if reference_voice:
                    similarity = cosine_similarity(reference_voice, audio_path)
                    if similarity is not None:
                        metrics["speaker_similarity"] = similarity

                results_writer.write_output(
                    run_id=run_id,
                    model_id=model_id_lookup[name],
                    prompt_id=prompt_id_lookup[prompt.id],
                    audio_path=str(audio_path),
                    sample_rate=sample_rate,
                    metrics=metrics,
                )
                outputs_payload.append(
                    {
                        "model": name,
                        "prompt_id": prompt.id,
                        "style": style,
                        "audio_path": str(audio_path),
                        "sample_rate": sample_rate,
                        "metrics": metrics,
                    }
                )

    results_payload = {
        "run": {
            "run_id": run_id,
            "created_at": run_info.created_at.isoformat(),
            "prompts_path": run_info.prompts_path,
            "seed": seed,
        },
        "models": model_rows,
        "prompts": prompt_rows,
        "outputs": outputs_payload,
    }
    results_writer.dump_json(run_dir / "results.json", results_payload)
    write_report(run_dir / "report.md", results_payload)
    console.print(f"Run complete: {run_dir}")


@app.command("benchmark")
def benchmark_cmd(
    models: str = typer.Option("all", help="Comma-separated model names or 'all'."),
    prompts: Path = typer.Option(..., help="Prompt YAML."),
    out: Path = typer.Option(Path("runs"), help="Output directory."),
    run_id: Optional[str] = typer.Option(None, help="Explicit run id."),
    seed: int = typer.Option(1337, help="Random seed."),
    reference_voice: Optional[Path] = typer.Option(None, help="Reference voice for similarity."),
) -> None:
    run_id = _run_id(run_id)

    if models == "all":
        selected_models = [model.name for model in list_models()]
    else:
        selected_models = [name.strip() for name in models.split(",") if name.strip()]

    _benchmark_run(
        models=selected_models,
        prompts=prompts,
        out=out,
        run_id=run_id,
        seed=seed,
        reference_voice=reference_voice,
    )


@app.command("train")
def train_cmd(
    recipe: str = typer.Option(..., help="xtts|vits|styletts2"),
    data: Path = typer.Option(..., help="Dataset directory."),
    out: Path = typer.Option(Path("training"), help="Output directory."),
    dry_run: bool = typer.Option(True, help="Only create plan/configs."),
    seed: int = typer.Option(1337, help="Random seed."),
) -> None:
    exp_id = _run_id(None)
    exp_dir = out / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = prepare_dataset(data, exp_dir / "prepared", seed=seed)
    plan = create_training_plan(recipe, dataset_info, exp_dir)
    (exp_dir / "training_plan.json").write_text(json.dumps(plan, indent=2))
    if dry_run:
        console.print("Dry-run complete. Review training_plan.json and plan.sh for next steps.")
        return
    console.print("Training execution is not implemented in this reference build.")


@app.command("eval-trained")
def eval_trained_cmd(
    checkpoint: Path = typer.Option(..., help="Model checkpoint to evaluate."),
    prompts: Path = typer.Option(..., help="Prompts YAML."),
    out: Path = typer.Option(Path("runs"), help="Output directory."),
    model: str = typer.Option("coqui_xtts_v2", help="Model name."),
    reference_voice: Optional[Path] = typer.Option(None, help="Reference voice for similarity."),
) -> None:
    run_id = _run_id(None)
    config_override: Dict[str, object] = {"model_path": str(checkpoint)}
    if reference_voice:
        config_override["speaker_wav"] = str(reference_voice)

    _benchmark_run(
        models=[model],
        prompts=prompts,
        out=out,
        run_id=run_id,
        seed=1337,
        reference_voice=reference_voice,
        config_override=config_override,
    )
