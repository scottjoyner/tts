from importlib.util import find_spec
from pathlib import Path

import pytest

from ttsbench.utils.prompts import load_prompts, normalize_prompt


def test_load_prompts(tmp_path: Path) -> None:
    if find_spec("yaml") is None:
        pytest.skip("pyyaml not available")
    content = """
    sample_rate: 22050
    prompts:
      - id: test
        text: " Hello   world "
        language: en
    """
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(content)

    prompt_set = load_prompts(prompt_file)
    assert prompt_set.config.sample_rate == 22050
    assert prompt_set.config.prompts[0].id == "test"
    assert normalize_prompt(prompt_set.config.prompts[0].text) == "Hello world"
