# TTSBench: Local-First TTS Benchmark + Voice Training

TTSBench provides a local-first benchmarking harness and training scaffolding for offline text-to-speech (TTS) workflows. It supports deterministic runs, structured logging, and repeatable artifacts (audio, metrics, SQLite, and Markdown reports).

## Features

- Benchmark multiple TTS models on shared prompts and settings.
- Structured JSON + SQLite results storage.
- Optional ASR-based intelligibility metrics.
- Audio metrics (duration, RMS, clipping %, optional LUFS).
- Training dataset preparation pipeline with train/val/test splits.
- Extensible plugin architecture for adding new models.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

List available models:

```bash
ttsbench list-models
```

Benchmark all models:

```bash
ttsbench benchmark --models all --prompts prompts.yaml --out runs/
```

Synthesize with a single model:

```bash
ttsbench synth --model piper --prompts prompts.yaml --out runs/
```

## Model Setup Notes

### Piper

- Install Piper and ensure `piper` is on `PATH`.
- Provide a local `.onnx` voice file by setting `voice` in `prompts.yaml`.

Example:

```bash
piper --model en_US-amy-medium.onnx --output_file out.wav < text.txt
```

### Coqui XTTS v2

- Install `TTS` from Coqui:
  ```bash
  pip install TTS
  ```
- Ensure XTTS v2 weights are available locally.
- Use `model_name` or `model_path` in `prompts.yaml`, and `speaker_wav` for voice cloning.

## Training (Single Speaker)

Prepare dataset and create training plan:

```bash
ttsbench train --recipe xtts --data /path/to/dataset --out training --dry-run
```

Dataset format:

- `metadata.csv` with columns: `audio_path,text`
- Or `metadata.jsonl` lines with `{ "audio_path": "relative/path.wav", "text": "..." }`

The command will:

- Validate audio files.
- Trim leading/trailing silence.
- Write train/val/test manifests.
- Produce `training_plan.json` and `plan.sh` with example commands.

## Results

Each benchmark run creates:

- `runs/<run_id>/results.json`
- `runs/<run_id>/results.sqlite`
- `runs/<run_id>/report.md`

## Offline Mode

Once models are downloaded and installed locally, all commands run offline. The benchmark pipeline avoids external APIs by default.

## Licensing

This repo does **not** download model weights automatically. Follow each model's licensing and download steps.

## Next Steps

See `docs/ADDING_A_MODEL.md` for guidance, including a Qwen TTS plugin stub roadmap.
