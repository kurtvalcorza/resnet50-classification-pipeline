from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Modern_Image_Classification_Workshop.ipynb"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def test_modern_image_classification_workshop_is_valid_notebook() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    markdown = "\n".join(_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown")
    assert "# DIMER Modern Image Classification Workshop" in markdown
    assert "DIMER Notebook Specification:** `2.1`" in markdown

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_modern_image_classification_workshop_carries_all_six_checkpoints() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    expected_model_ids = (
        "timm/resnet50.a1_in1k",
        "timm/mobilenetv4_conv_small.e2400_r224_in1k",
        "timm/convnext_tiny.in12k_ft_in1k",
        "timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        "timm/swinv2_tiny_window8_256.ms_in1k",
        "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    )
    for model_id in expected_model_ids:
        assert model_id in text
