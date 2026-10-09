"""Regression tests for the resnet50_classification_colab review fixes (RN-M1..M5, RN-m1..m6).

Ported from the sibling convnext-classification-pipeline (06482ea, same generator lineage).

They run within CI's install budget (CPU torch, timm, Pillow, NumPy; no model weights): the fine-tuning helpers are
exercised on a tiny random-initialised timm ResNet (resnet10t, with BatchNorm) stand-in, and the notebook's own Section 8-10 cells are executed
with stub pipelines. None of this is pretrained-inference or clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

with contextlib.suppress(ImportError):  # load torch before any NumPy BLAS call (Windows DLL load-order trap)
    import torch  # noqa: F401

from PIL import Image

from resnet50_classification_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "resnet50_classification_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _png(color: tuple[int, int, int], size: tuple[int, int] = (24, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


GOOD = {
    f"{c}/{k}.png": _png(col) for c, col in (("red", (200, 0, 0)), ("blue", (0, 0, 200))) for k in range(5)
}


# --- RN-M1: isolated runtime, no in-kernel install ------------------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install = kernel[0]
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in kernel[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "timm==1.0.29", "numpy==2.5.3", "pillow==11.3.0"):
        assert pin in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """accelerate-style find_spec("google.colab") must not raise on the worker's stubs (fleet Colab failure)."""
    import importlib.util

    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        shim_globals = {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None}
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(
            sys.modules["google.colab.files"].upload
        )
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_release_record_no_longer_counts_the_restarted_run_as_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    # RN-m5: the 2026-09-14 Kaggle row of the previous blob is no longer a PASS (the workshop's own rows stay).
    row = next(line for line in text.splitlines() if line.startswith("| 2026-09-14 | `b7e787d`"))
    assert "Restart status not recorded; not promotion evidence." in row and "**PASSED**" not in row
    assert "No hosted run of the current notebook blob is recorded yet" in text
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate" in status


# --- RN-M3 / RN-m3: trainable set and configuration ---------------------------------------------------------------------


def test_head_only_counts_are_the_classifier_and_full_counts_everything() -> None:
    pytest.importorskip("timm")
    head = pl.trainable_parameter_counts(2, train_backbone=False)
    full = pl.trainable_parameter_counts(2, train_backbone=True)
    assert head["trainable_parameters"] == 2048 * 2 + 2 and head["method"] == "head-only fine-tuning"
    assert full["frozen_parameters"] == 0 and full["trainable"] == "all"
    assert (
        full["trainable_parameters"] == head["trainable_parameters"] + head["frozen_parameters"] == 23_512_130
    )
    with pytest.raises(TypeError):
        pl.trainable_parameter_counts(2, train_backbone="head")


@pytest.fixture()
def tiny_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`fit` on a random-initialised resnet10t stand-in (BatchNorm, ``fc`` head; no weights read); returns a runner."""
    timm = pytest.importorskip("timm")
    real_create = timm.create_model

    def create(
        name: str, pretrained: bool = False, pretrained_cfg_overlay: Any = None, num_classes: int = 1000
    ):
        return real_create("resnet10t", pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(timm, "create_model", create)
    weights = tmp_path / "weights"
    weights.mkdir()
    images = [
        Image.new("RGB", (40, 40), (200 if k % 2 else 0, 60, 200 if k % 2 == 0 else 0)) for k in range(8)
    ]
    targets = [k % 2 for k in range(8)]

    def run(out: str, **overrides: Any):
        kwargs = {
            "epochs": 1,
            "batch_size": 2,
            "weights_dir": weights,
            "output_dir": tmp_path / out,
            "allow_download": True,
            "provenance": {"source": "test", "sha256": "0" * 64},
            **overrides,
        }
        return pl.ResNet50ClassificationPipeline.fit(
            images[:6], targets[:6], images[6:], targets[6:], ["a", "b"], **kwargs
        )

    return run, tmp_path


def test_fit_head_only_moves_only_the_head_and_records_its_configuration(tiny_fit) -> None:
    import torch
    from safetensors.torch import load_file

    run, tmp_path = tiny_fit
    torch.manual_seed(0)
    _pipe, meta = run("head", train_backbone=False, seed=7)
    config = meta["config"]
    assert config["method"] == "head-only fine-tuning" and config["trainable"] == "head"
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "seed",
        "trainable_parameters",
        "frozen_parameters",
        "train_samples",
        "val_samples",
        "dataset",
    ):
        assert key in config, key
    saved = json.loads((tmp_path / "head" / "model-config.json").read_text(encoding="utf-8"))
    assert saved["fine_tuning"] == json.loads(json.dumps(config))
    assert saved["fine_tuning"]["dataset"]["sha256"] == "0" * 64
    # Re-create the same initial model (same seed, same creation order) and compare every tensor, BatchNorm running
    # statistics included (RN-M3): only the new fc head moved.
    import timm

    torch.manual_seed(7)
    initial = timm.create_model("resnet10t", num_classes=2).state_dict()
    tuned = load_file(str(tmp_path / "head" / "model.safetensors"))
    assert any("running_mean" in k for k in tuned)
    changed = {k for k, v in tuned.items() if not torch.equal(v, initial[k])}
    assert changed == {"fc.weight", "fc.bias"}
    assert config["batchnorm_statistics"] == "frozen (eval mode)"


def test_fit_is_repeatable_with_the_same_seed_and_rejects_bad_settings(tiny_fit) -> None:
    run, _tmp = tiny_fit
    _a, first = run("one", seed=11)
    _b, second = run("two", seed=11)
    assert first["history"] == second["history"]
    assert first["config"]["method"] == "full fine-tuning" and first["config"]["frozen_parameters"] == 0
    for bad in ({"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0.0}, {"weight_decay": -1.0}):
        with pytest.raises(ValueError):
            run("bad", **bad)


# --- RN-M2: verdict from the counts, with its uncertainty -------------------------------------------------------


def test_wilson_interval_values() -> None:
    low, high = pl.wilson_interval(5, 6)
    assert (round(low, 3), round(high, 3)) == (0.436, 0.97)
    low, high = pl.wilson_interval(6, 6)
    assert round(low, 3) == 0.61 and high == 1.0
    assert pl.wilson_interval(0, 6)[0] == 0.0
    with pytest.raises(ValueError):
        pl.wilson_interval(7, 6)


@pytest.mark.parametrize(
    ("predicted", "comparison"),
    [
        ([0, 0, 0, 1, 1, 0], "indistinguishable-from-baseline"),  # 5/6, interval contains 0.5
        ([0, 0, 0, 0, 0, 0], "indistinguishable-from-baseline"),  # 3/6 = the baseline
        ([0, 0, 0, 1, 1, 1], "above-baseline"),  # 6/6
    ],
)
def test_finetune_report_never_claims_success(predicted: list[int], comparison: str) -> None:
    truth = [0, 0, 0, 1, 1, 1]
    report = pl.finetune_evaluation_report(
        truth, predicted, ["frog", "truck"], item_ids=[f"i{k}" for k in range(6)]
    )
    assert report["verdict"] == "sample-sanity"
    assert report["comparison_to_baseline"] == comparison
    assert report["n"] == 6 and report["metrics"]["majority_baseline_accuracy"] == 0.5
    assert "not a benchmark" in report["estimation"]
    assert len(report["misclassified"]) == 6 - report["correct"]
    assert sum(c["correct"] for c in report["per_class"].values()) == report["correct"]


def test_below_baseline_when_the_interval_lies_under_it() -> None:
    truth = [0] * 18 + [1] * 2
    report = pl.finetune_evaluation_report(truth, [1] * 20, ["a", "b"])
    assert report["comparison_to_baseline"] == "below-baseline"


def test_notebook_has_no_literal_verdict_and_interprets_the_fine_tune(nb: dict) -> None:
    code = "\n".join(_src(c) for c in _code_cells(nb))
    assert "'verdict': 'success'" not in code
    markdown = _markdown(nb)
    closing = markdown[markdown.index("## Interpretation and limits") :]
    assert "**40 images**" in closing and "95 % interval" in closing


# --- RN-m1 / RN-M5 / RN-m4: data intake, fallback, reload equivalence -----------------------------------------


def test_load_image_zip_reproduces_the_previous_split_when_no_names_repeat() -> None:
    """With no shared file names every group is one image, so the split equals the previous per-image one."""
    import random

    entries = {
        f"set/{c}/image_{k}.png": _png((k * 9 % 255, 30, 90 if c == "frog" else 10))
        for c in ("frog", "truck")
        for k in range(24)
    }
    data = pl.load_image_zip(_zip(entries), seed=42, validation_split=0.2, subset_per_class=16)
    # The notebook's previous inline algorithm: classes in archive order, one Random(seed) across classes.
    by_class: dict[str, list[str]] = {}
    for name in entries:
        by_class.setdefault(name.split("/")[-2], []).append(name)
    rng = random.Random(42)
    expected_train, expected_val = [], []
    for files in by_class.values():
        files = list(files)
        rng.shuffle(files)
        files = files[:16]
        n_val = max(1, int(len(files) * 0.2))
        expected_val += files[:n_val]
        expected_train += files[n_val:]
    assert data["train_ids"] == expected_train and data["val_ids"] == expected_val
    assert data["classes"] == ["frog", "truck"] and data["manifest"]["per_class"]["frog"] == {"train": 13, "val": 3}


def _pair_archive() -> bytes:
    """The tutorial archive's layout: 2 classes x 100 photos, each stored as original_images/ and darkened_images/."""
    return _zip(
        {
            f"CIFAR-10-subset/{v}/{c}/image_{k}.png": _png((k % 256, 0 if v == "original_images" else 9, 50 if c == "frog" else 0), (4, 4))
            for v in ("original_images", "darkened_images")
            for c in ("frog", "truck")
            for k in range(100)
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_pair_grouped_split_never_straddles_a_photo(seed: int) -> None:
    """RN-M2 (Kurt's convnext CNX-M3 option B): SUBSET_PER_CLASS=100 keeps 50 photo pairs per class; no pair straddles train/held-out."""
    import os

    data = pl.load_image_zip(_pair_archive(), seed=seed, validation_split=0.2, subset_per_class=100)
    assert data["manifest"]["per_class"] == {"frog": {"train": 80, "val": 20}, "truck": {"train": 80, "val": 20}}
    train = {(t, os.path.basename(i)) for t, i in zip(data["train_targets"], data["train_ids"], strict=True)}
    val = [(t, os.path.basename(i)) for t, i in zip(data["val_targets"], data["val_ids"], strict=True)]
    assert not train.intersection(val)
    # every kept photo appears with both copies on its side
    for side in (data["train_ids"], data["val_ids"]):
        names = [i.replace("darkened_images", "original_images") for i in side]
        assert all(names.count(n) == 2 for n in names)


def test_notebook_default_is_head_only_and_section_12_is_full_fine_tuning(nb: dict) -> None:
    """Kurt 2026-10-04 (convnext CNX-M2, applied to RN-M3): head-only by default; full fine-tuning is the activity."""
    fit_cell = _cell(nb, "TRAINABLE = 'head'")
    assert re.search(r"^TRAINABLE = 'head'  # @param", fit_cell, re.M)
    assert "train_backbone = TRAINABLE == 'all'" in fit_cell
    markdown = _markdown(nb)
    section12 = markdown[markdown.index("## 12. Your turn") :]
    assert "TRAINABLE = 'all'" in section12 and "BatchNorm" in section12
    assert "Set TRAINABLE = 'all' in Section 9" in _cell(nb, "run_history:")


def test_notebook_default_uses_the_grouped_subset_of_100(nb: dict) -> None:
    data_cell = _cell(nb, "USE_BYOD_DATASET = False")
    assert re.search(r"^SUBSET_PER_CLASS = 100\b", data_cell, re.M)
    assert "subset_per_class=SUBSET_PER_CLASS" in data_cell


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            {
                "train/red/1.png": _png((200, 0, 0)),
                "train/blue/1.png": _png((0, 0, 200)),
                "val/red/2.png": _png((200, 0, 0)),
                "val/green/2.png": _png((0, 200, 0)),
            },
            r"\['green'\] appear in val/",
        ),
        ({**GOOD, "stray.png": _png((0, 200, 0))}, r"stray\.png: image is not inside a class folder"),
        ({**GOOD, "red/huge.png": _png((200, 0, 0), (4100, 8))}, r"red/huge\.png: image side outside"),
        ({**GOOD, "blue/broken.png": b"not an image"}, r"blue/broken\.png: not a decodable image"),
        ({f"red/{k}.png": _png((200, 0, 0)) for k in range(4)}, r"at least 2 classes"),
        ({**GOOD, "green/1.png": _png((0, 200, 0))}, r"class 'green' has 1 image"),
        ({**GOOD, "../evil.png": _png((0, 0, 0))}, r"illegal path"),
        ({"notes.txt": b"x"}, r"no image files"),
        # RN-M5: a user's test/ split is refused, never pooled into training.
        (
            {**{f"train/{k}": v for k, v in GOOD.items()}, **{f"test/{k}": v for k, v in GOOD.items()}},
            r"test/.*has a test/ folder",
        ),
        ({**GOOD, "Testing/red/9.png": _png((200, 0, 0))}, r"has a test/ folder"),
        # RN-M5: a class present in train/ but absent from val/ cannot be measured.
        (
            {
                "train/red/1.png": _png((200, 0, 0)),
                "train/blue/1.png": _png((0, 0, 200)),
                "val/red/2.png": _png((200, 0, 0)),
            },
            r"\['blue'\] appear in train/ but not in val/",
        ),
    ],
)
def test_load_image_zip_refuses_bad_archives_before_training(entries: dict[str, bytes], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pl.load_image_zip(_zip(entries), seed=0)


def test_load_image_zip_limits_and_skipped_files(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pl.load_image_zip(_zip({**GOOD, "readme.txt": b"hi", "__MACOSX/red/._1.png": b"x"}), seed=0)
    assert {s["file"] for s in data["manifest"]["skipped"]} == {"readme.txt", "__MACOSX/red/._1.png"}
    monkeypatch.setattr(pl, "MAX_DATASET_IMAGES", 5)
    with pytest.raises(ValueError, match="MAX_DATASET_IMAGES=5"):
        pl.load_image_zip(_zip(GOOD), seed=0)
    monkeypatch.setattr(pl, "MAX_DATASET_ARCHIVE_BYTES", 10)
    with pytest.raises(ValueError, match="MAX_DATASET_ARCHIVE_BYTES=10"):
        pl.load_image_zip(_zip(GOOD), seed=0)


def test_presplit_archive_is_used_as_given() -> None:
    entries = {
        "train/red/1.png": _png((200, 0, 0)),
        "train/blue/1.png": _png((0, 0, 200)),
        "val/red/2.png": _png((190, 0, 0)),
        "val/blue/2.png": _png((0, 0, 190)),
    }
    data = pl.load_image_zip(_zip(entries), seed=0)
    assert sorted(data["val_ids"]) == ["val/blue/2.png", "val/red/2.png"]
    assert data["manifest"]["layout"] == "train/ and val/ folders"


class _StubPipe:
    """Stands in for a fine-tuned pipeline: predicts class 0 for red-dominant images, else class 1."""

    def __init__(self, labels: list[str], flip: bool = False) -> None:
        self.labels, self.flip, self.source, self.device = labels, flip, "fine-tuned-artifact", "cpu"

    def predict(self, image: Image.Image) -> dict:
        red, _green, blue = image.convert("RGB").getpixel((0, 0))
        index = 0 if red >= blue else 1
        if self.flip:
            index = 1 - index
        top = [
            {"index": index, "label": self.labels[index], "score": 0.9},
            {"index": 1 - index, "label": self.labels[1 - index], "score": 0.1},
        ]
        return {
            "predictions": [{"predicted_index": index, "predicted_label": self.labels[index], "top_k": top}]
        }


def _stub_namespace(tmp_path: Path, reload_flip: bool = False) -> dict[str, Any]:
    import os

    ns: dict[str, Any] = {k: v for k, v in vars(pl).items() if not k.startswith("__")}
    ns["__name__"] = "__main__"

    class Pipe:
        @staticmethod
        def fit(
            train_images,
            train_targets,
            val_images,
            val_targets,
            class_names,
            *,
            output_dir,
            train_backbone,
            provenance,
            **kwargs,
        ):
            os.makedirs(output_dir, exist_ok=True)
            for name in ("model.safetensors", "model-config.json"):
                Path(output_dir, name).write_text("{}", encoding="utf-8")
            config = {
                "method": "full fine-tuning" if train_backbone else "head-only fine-tuning",
                "trainable_parameters": 23_512_130 if train_backbone else 4098,
                "frozen_parameters": 0 if train_backbone else 23_508_032,
                "train_samples": len(train_targets),
                "val_samples": len(val_targets),
                "dataset": provenance,
                **{k: kwargs[k] for k in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed")},
            }
            history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_accuracy": 0.5}]
            return _StubPipe(list(class_names)), {"history": history, "device": "cpu", "config": config}

    class Reloader:
        @staticmethod
        def from_pretrained(weights_dir):
            return _StubPipe(ns["CUSTOM_CLASSES"], flip=reload_flip)

    ns.update(
        pipe=Pipe(),
        ResNet50ClassificationPipeline=Reloader,
        WEIGHTS_DIR=tmp_path / "weights",
        trainable_parameter_counts=lambda n, train_backbone: {
            "method": "full fine-tuning" if train_backbone else "head-only fine-tuning",
            "trainable_parameters": 23_512_130 if train_backbone else 4098,
            "frozen_parameters": 0 if train_backbone else 23_508_032,
        },
    )
    return ns


def _run(source: str, ns: dict[str, Any]) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "cell", "exec"), ns)
    return out.getvalue()


def test_air_gapped_fallback_runs_through_fine_tuning_and_evaluation(
    nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RN-m1: the download fails, the fallback dataset is built, and Sections 9-10 run on it."""
    import urllib.request

    def offline(*args: Any, **kwargs: Any):
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns = _stub_namespace(tmp_path)
    printed = _run(_cell(nb, "USE_BYOD_DATASET = False"), ns)
    assert "falling back to the deterministic synthetic stripes dataset" in printed
    assert ns["dataset_kind"] == "synthetic fallback" and len(ns["val_images"]) == 4
    assert all(img.size == (64, 64) for img in ns["train_images"] + ns["val_images"])
    _run(_cell(nb, "TRAINABLE = 'head'"), ns)
    printed = _run(_cell(nb, "EQUIVALENCE_TOLERANCE"), ns)
    report = ns["finetuned_eval_report"]
    assert report["sample_kind"] == "synthetic fallback" and report["equivalence"]["equivalent"] is True
    assert report["verdict"] == "sample-sanity" and "Wilson interval" in printed
    assert set(report["artifact_sha256"]) == {"model.safetensors", "model-config.json"}
    assert len(ns["run_history"]) == 1
    assert (tmp_path / "outputs" / "resnet50_classification_finetuned_evaluation_report.json").is_file()


def test_byod_archive_flows_through_the_cell_and_a_bad_one_stops_before_training(
    nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_cell = re.sub(
        r"^USE_BYOD_DATASET = False",
        "USE_BYOD_DATASET = True",
        _cell(nb, "USE_BYOD_DATASET = False"),
        count=1,
        flags=re.M,
    )
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    good.write_bytes(_zip(GOOD))
    bad.write_bytes(_zip({**GOOD, "stray.png": _png((1, 2, 3))}))
    ns = _stub_namespace(tmp_path)
    _run(data_cell.replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {str(good)!r}"), ns)
    assert ns["dataset_kind"] == "BYOD" and ns["CUSTOM_CLASSES"] == ["blue", "red"]
    ns = _stub_namespace(tmp_path)
    with pytest.raises(ValueError, match="stray.png"):
        _run(data_cell.replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {str(bad)!r}"), ns)
    assert "fine_tuned_pipe" not in ns


def test_reload_cell_stops_when_the_artifact_does_not_reproduce_the_model(
    nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RN-m4: loading is not enough; a reloaded model that predicts differently is refused."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns = _stub_namespace(tmp_path, reload_flip=True)
    archive = tmp_path / "good.zip"
    archive.write_bytes(_zip(GOOD))
    data_cell = _cell(nb, "USE_BYOD_DATASET = False").replace(
        "USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1
    )
    _run(data_cell.replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {str(archive)!r}"), ns)
    _run(_cell(nb, "TRAINABLE = 'head'"), ns)
    with pytest.raises(RuntimeError, match="does not reproduce the in-memory model"):
        _run(_cell(nb, "EQUIVALENCE_TOLERANCE"), ns)


def test_byod_image_that_is_not_an_image_is_named(nb: dict, tmp_path: Path) -> None:
    sample = _cell(nb, "USE_BYOD = False")
    bogus = tmp_path / "notes.png"
    bogus.write_bytes(b"this is text")
    source = sample.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace(
        "BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {str(bogus)!r}"
    )
    ns: dict[str, Any] = {"NUM_CLASSES": 1000, "__name__": "__main__"}
    with pytest.raises(ValueError, match=r"notes\.png: not a decodable image"):
        _run(source, ns)


# --- RN-M4 / RN-m5 / RN-m6: guided layer and corrected statements ------------------------------------------------------


def test_guided_layer_and_infrastructure_cells(nb: dict) -> None:
    markdown = _markdown(nb)
    for marker in (
        "**Who this is for.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "**Input → Model → Output.**",
        "## 12. Your turn — change one thing",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
    ):
        assert marker in markdown, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 5 and all(c["metadata"].get("cellView") == "form" for c in titled)
    fit_cell = _cell(nb, "TRAINABLE = 'head'")
    for field in ("TRAINABLE", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "TRAIN_SEED"):
        assert re.search(rf"^{field} = .*# @param", fit_cell, re.M), field
    for stale in ("classification head fine-tuning", "nothing is downloaded", "gracefully falls back"):
        assert stale not in markdown
    assert "986,707-byte" in markdown and "Runtime → Run after" in markdown


def test_interval_folder_is_a_class_folder_not_a_val_split() -> None:
    """RN-M5: split folders match whole path components, so `interval/` no longer empties the validation set."""
    entries = {f"interval/{k}": v for k, v in GOOD.items()}
    data = pl.load_image_zip(_zip(entries), seed=0)
    assert data["manifest"]["layout"].startswith("class folders")
    assert data["val_images"] and data["train_images"]


def test_dataset_digest_mismatch_stops_naming_both_digests(
    nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RN-m1: an altered download is refused (fail closed), never routed into the synthetic fallback."""
    import urllib.request

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(_zip(GOOD)))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns = _stub_namespace(tmp_path)
    with pytest.raises(ValueError, match=r"has SHA-256 [0-9a-f]{64}, expected 66f90a4f") as caught:
        _run(_cell(nb, "USE_BYOD_DATASET = False"), ns)
    assert "SAMPLE_HEIGHT" not in str(caught.value)
    assert "dataset_kind" not in ns
