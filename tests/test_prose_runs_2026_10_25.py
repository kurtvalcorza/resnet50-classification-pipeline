"""Prose numbers name the run they came from (Kurt, 2026-10-25: point to output + ranges).

The SPLIT_SEED 0/1/2 scores (34/40, 37/40, 32/40) come from the local Windows CPU check of 2026-10-09; the
hosted Colab T4 runs (`75e7e1ee` 2026-10-09, `1c89a8be` 2026-10-25) ran the default split only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "resnet50_classification_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_split_seed_scores_are_credited_to_the_cpu_check_everywhere():
    md = _markdown()
    assert "with `SPLIT_SEED` 0, 1 and 2 it scored" not in md
    assert md.count("34/40, 37/40 and 32/40") == 3
    for chunk in md.split("34/40, 37/40 and 32/40")[:-1]:
        tail = chunk[-160:]
        assert "CPU check" in tail and "not" in tail and "hosted" in tail, tail
