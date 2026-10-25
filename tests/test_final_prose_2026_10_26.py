"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
resnet50 (Kurt's ruling): the SPLIT_SEED 0/1/2 scores are dropped;
the split-seed check is cited by commit 72a6cda.
Fails on PR head e7ad649.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/resnet50_classification_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    '37/40',
    '32/40',
    'A GPU run can differ by an image or two',
    'seeds or devices move the count',
    'consistent evidence',
]
PRESENT = [
    'a local CPU split-seed check (not a hosted run; commit `72a6cda`)',
    'The count can change with the split',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
