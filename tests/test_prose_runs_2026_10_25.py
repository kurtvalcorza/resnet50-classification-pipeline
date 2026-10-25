"""Prose numbers name the run they came from (Kurt, 2026-10-25: point to output + ranges).

The SPLIT_SEED 0/1/2 scores were dropped from the prose (Kurt, 2026-10-26); the local CPU split-seed check is cited by commit 72a6cda; the
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
    assert "37/40" not in md and "32/40" not in md
    assert md.count("local CPU split-seed check") == 3
    assert md.count("commit `72a6cda`) ") >= 2


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): every quoted hosted-run number names all three Colab T4 runs,
    # timings are records, and the unsupported "add a few minutes" estimate is gone.
    # Fails on PR #15 head 0ad859d.
    md = _markdown()
    for gone in ("the hosted Colab T4 run of blob `75e7e1ee`", "add a few minutes (an estimate)",
                 "the same day"):
        assert gone not in md, gone
    for quoted in ("83.7 s (`1c89a8be`, commit `e5f1e56`, 25 October 2026; setup 47 s)",
                   "`8cb6ad08` (commit `3a1afba`, both 25 October 2026), which printed the same numbers",
                   "your run may take a different time"):
        assert quoted in md, quoted
