# Adding a TTS Model Plugin

1. Create a new plugin in `ttsbench/models/plugins/<name>.py` that subclasses `BaseTTSModel`.
2. Implement:
   - `is_available()` to detect local installation.
   - `availability_help()` for setup instructions.
   - `synth()` that writes audio to `out_dir` and returns timings + stats.
3. Register the model in `ttsbench/models/registry.py`.
4. Update `README.md` with install/config notes.

## Qwen TTS Next Steps

Once the exact Qwen TTS repository/checkpoint is known:

- Add a `QwenTTSModel` implementation in `ttsbench/models/plugins/qwen_tts.py` that loads the local checkpoint.
- Wire configuration keys like `model_path`, `speaker_wav`, `language`, and `style` to the Qwen API.
- Update `availability_help()` with the expected local file layout and any licensing constraints.
- Register the plugin and add a short section to the README describing setup.
