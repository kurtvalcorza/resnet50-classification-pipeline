"""Acceptance checks for the Notebook Review Framework v1 findings on the modern image workshop.

The tests execute the notebook's own code cells. A tiny fake backbone (a fixed pixel-to-feature map) stands in
for the pretrained checkpoints, so no weights or photos are downloaded; the probe fitting, SafeTensors export,
artifact reconstruction, 5-NN rule, metric validation, BYOD loader and activity code are the notebook's own.
These are logic checks, not model runs or Colab evidence.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Modern_Image_Classification_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"]}
MARKDOWN = "\n".join(CELLS[c["id"]] for c in NB["cells"] if c["cell_type"] == "markdown")


class Recorder:
    """A stand-in for matplotlib.pyplot: records shown figures and every text drawn."""

    def __init__(self):
        self.shown = 0
        self.texts = []
        self.images = 0

    def subplots(self, rows=1, cols=1, squeeze=True, **kwargs):
        grid = np.empty((rows, cols), dtype=object)
        for index in np.ndindex(rows, cols):
            grid[index] = self
        return self, (grid if not squeeze or grid.size > 1 else grid[0, 0])

    def figure(self, *args, **kwargs):
        return self

    def add_subplot(self, *args, **kwargs):
        return self

    def show(self, *args, **kwargs):
        self.shown += 1

    def imshow(self, *args, **kwargs):
        self.images += 1

    def text(self, x, y, s, *args, **kwargs):
        self.texts.append(str(s))

    def set_title(self, s, *args, **kwargs):
        self.texts.append(str(s))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def run(cell_id, ns, cut=None, replacements=None):
    text = CELLS[cell_id]
    if cut:
        text = text[: text.index(cut)]
    for old, new in (replacements or {}).items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    exec(compile(text, cell_id, "exec"), ns)
    return ns


@pytest.fixture
def ns(tmp_path, monkeypatch):
    """The notebook's configuration, dataset constants, model registry and methods, without network access."""
    monkeypatch.chdir(tmp_path)
    plt = Recorder()
    namespace = {
        "display": print,
        "plt": plt,
        "np": np,
        "torch": torch,
        "F": torch.nn.functional,
        "Image": Image,
    }
    from safetensors.torch import load_file, save_file

    namespace.update(load_file=load_file, save_file=save_file)
    run("4f54b044", namespace)
    run("e04f3109", namespace, cut="records = fetch_dataset()")
    run("f470276e", namespace)
    run("f370927c", namespace)
    run("6e025182", namespace, cut="ALL_RESULTS={}")
    run("22e6f1e2", namespace, cut="model_metric_rows=[]")
    namespace["RUNTIME"] = {"python": "test"}
    namespace["DEVICE"] = "cpu"
    return namespace


def colour_image(label_id, index, size=(40, 32)):
    colour = (60 * label_id % 256, (30 + index) % 256, (200 - 50 * label_id) % 256)
    return Image.new("RGB", (size[0] + index % 7, size[1]), colour)


class FakeBackbone:
    """forward_features/forward_head over a fixed 8-number summary of each image."""

    def forward_features(self, x):
        return x

    def forward_head(self, x, pre_logits=True):
        return x


def fake_transform(seen_sizes):
    def transform(image):
        seen_sizes.append(image.size)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        summary = [
            *pixels.mean(axis=(0, 1)),
            *pixels.std(axis=(0, 1)),
            image.size[0] / 100,
            image.size[1] / 100,
        ]
        return torch.tensor(summary, dtype=torch.float32)

    return transform


# --- M1: 5-NN is scored by its own tie rule


def knn_case(ns, sims, labels, n_classes=6):
    """Build unit vectors whose cosine similarity to one query is exactly `sims`."""
    sims = np.asarray(sims, dtype=float)
    train = np.stack(
        [np.r_[s, np.sqrt(max(0.0, 1 - s * s)) * np.eye(len(sims))[i]] for i, s in enumerate(sims)]
    )
    query = np.r_[1.0, np.zeros(len(sims))][None, :]
    return ns["knn_predict"](train, np.asarray(labels), query, 5, n_classes=n_classes)


@pytest.mark.parametrize(
    ("sims", "labels", "expected"),
    [
        (
            [0.99, 0.98, 0.80, 0.70, 0.60],
            [1, 1, 0, 0, 2],
            1,
        ),  # 2-2-1 tie, class 1 has the larger similarity sum
        (
            [0.99, 0.60, 0.80, 0.70, 0.98],
            [0, 0, 1, 1, 2],
            0,
        ),  # 2-2-1 tie: class 0 sums 1.59, class 1 sums 1.50
        (
            [0.90, 0.80, 0.70, 0.60, 0.50],
            [4, 3, 2, 1, 0],
            4,
        ),  # five-way tie: the most similar neighbour's class
        (
            [0.90, 0.60, 0.80, 0.70, 0.50],
            [3, 3, 2, 2, 5],
            2,
        ),  # equal summed similarity (1.5 each): lower class id
        ([0.90, 0.80, 0.70, 0.60, 0.50], [5, 5, 5, 0, 0], 5),  # unique majority
    ],
)
def test_knn_decision_follows_votes_then_similarity_then_class_id(ns, sims, labels, expected):
    pred, probs = knn_case(ns, sims, labels)
    assert int(pred[0]) == expected
    votes = np.bincount(labels, minlength=6) / 5
    np.testing.assert_allclose(probs[0], votes)  # the scores stay plain vote fractions
    truth = np.array([expected])
    metrics = ns["classification_metrics"](truth, probs, predicted_ids=pred)
    assert metrics["accuracy"] == 1.0
    assert list(metrics["predicted_ids"]) == [expected]


def test_the_reported_tie_is_scored_by_the_decision_not_by_argmax(ns):
    pred, probs = knn_case(ns, [0.99, 0.98, 0.80, 0.70, 0.60], [1, 1, 0, 0, 2])
    assert int(probs.argmax(1)[0]) == 0  # what the evaluator used to score
    assert ns["classification_metrics"](np.array([1]), probs, predicted_ids=pred)["accuracy"] == 1.0


def test_default_and_byod_call_sites_pass_the_knn_decision():
    assert 'classification_metrics(te["labels"],knn_probs,predicted_ids=knn_pred)' in CELLS["6e025182"]
    assert (
        'classification_metrics(te["labels"],knn_probs,byod_labels,predicted_ids=knn_pred)'
        in CELLS["71cf319a"]
    )
    assert "def generic_knn_predict" not in CELLS["71cf319a"]


@pytest.mark.parametrize(
    ("probs", "message"),
    [
        (np.full((2, 6), 2.0), "between 0 and 1"),
        (np.full((2, 6), 0.1), "sum to 1"),
        (np.array([[np.nan] * 6, [1 / 6] * 6]), "non-finite"),
        (np.full((2, 5), 0.2), "shape"),
    ],
)
def test_invalid_scores_are_rejected_before_metrics(ns, probs, message):
    with pytest.raises(ValueError, match=message):
        ns["classification_metrics"](np.array([0, 1]), probs)


def test_out_of_range_class_ids_are_rejected(ns):
    probs = np.full((2, 6), 1 / 6)
    with pytest.raises(ValueError, match="class ids"):
        ns["classification_metrics"](np.array([0, 7]), probs)
    with pytest.raises(ValueError, match="class ids"):
        ns["classification_metrics"](np.array([0, 1]), probs, predicted_ids=np.array([0, 9]))


# --- M6: the probe is rebuilt from its artifact directory and its labels are checked


@pytest.fixture
def artifact(ns):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(6, 16)) * 3

    def draw(n):
        y = np.repeat(np.arange(6), n)
        return (centres[y] + rng.normal(size=(len(y), 16))).astype(np.float32), y

    (tr, ytr), (va, yva), (te, _yte) = draw(6), draw(2), draw(2)
    probe = ns["fit_probe"](tr, ytr, va, yva, te)
    spec = ns["MODEL_REGISTRY"]["resnet50"]
    path, _manifest = ns["save_probe_artifact"]("resnet50", spec, probe, 16, "train", "validation")
    return {"dir": path.parent, "spec": spec, "test_x": te, "probs": probe["test_probs"]}


def test_unchanged_artifact_reloads_with_identical_probabilities_and_labels(ns, artifact):
    diff = ns["reload_probe_verify"](
        artifact["dir"], artifact["spec"], artifact["test_x"], artifact["probs"], ns["CLASS_KEYS"]
    )
    assert diff == 0.0


def _edit_manifest(directory, change):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


MUTATIONS = {
    "reversed_class_order": (
        lambda d: _edit_manifest(d, lambda m: m.update(class_order=m["class_order"][::-1])),
        "class_order differs",
    ),
    "false_probe_digest": (
        lambda d: _edit_manifest(d, lambda m: m["probe_file"].update(sha256="0" * 64)),
        "does not match the digest",
    ),
    "changed_probe_bytes": (
        lambda d: (d / "probe.safetensors").open("ab").write(b"\0"),
        "does not match the digest",
    ),
    "missing_manifest": (lambda d: (d / "manifest.json").unlink(), "manifest.json is missing"),
    "wrong_dimension": (lambda d: _edit_manifest(d, lambda m: m.update(feature_dim=15)), "must have shape"),
    "wrong_base_revision": (
        lambda d: _edit_manifest(d, lambda m: m.update(base_model_revision="x")),
        "identity",
    ),
    "duplicate_classes": (lambda d: _edit_manifest(d, lambda m: m.update(class_order=["a"] * 6)), "distinct"),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_changed_or_inconsistent_artifacts_are_rejected(ns, artifact, mutation):
    change, message = MUTATIONS[mutation]
    change(artifact["dir"])
    with pytest.raises(RuntimeError, match=message):
        ns["reload_probe_verify"](
            artifact["dir"], artifact["spec"], artifact["test_x"], artifact["probs"], ns["CLASS_KEYS"]
        )


def test_reconstruction_uses_only_the_artifact_directory(ns, artifact):
    probe = ns["load_probe_artifact"](artifact["dir"], artifact["spec"])
    probs, labels = ns["probe_predict"](probe, artifact["test_x"])
    np.testing.assert_array_equal(probs, artifact["probs"])
    assert labels == [ns["CLASS_KEYS"][i] for i in artifact["probs"].argmax(1)]


# --- M3: resolution stress runs, reuses the saved probe, and never refits


def stress_namespace(ns, tmp_path, enabled):
    keys = ["resnet50", "vit"]
    ns["MODEL_REGISTRY"] = {k: ns["MODEL_REGISTRY"][k] for k in keys}
    seen = []
    transform = fake_transform(seen)
    ns["load_backbone"] = lambda key: (FakeBackbone(), transform, {}, None, 0.0, 0.0)
    records = [
        {
            "id": f"test-{i:03d}",
            "source_id": f"s{i}",
            "label_id": i % 6,
            "image": colour_image(i % 6, i, size=(200, 150)),
        }
        for i in range(24)
    ]
    train = [
        {
            "id": f"train-{i:03d}",
            "source_id": f"t{i}",
            "label_id": i % 6,
            "image": colour_image(i % 6, i + 50),
        }
        for i in range(36)
    ]
    ns["splits"] = {"test": records, "train": train}
    for key, spec in ns["MODEL_REGISTRY"].items():
        tr = ns["extract_features"](FakeBackbone(), transform, train)
        te = ns["extract_features"](FakeBackbone(), transform, records)
        probe = ns["fit_probe"](tr["features"], tr["labels"], tr["features"], tr["labels"], te["features"])
        ns["save_probe_artifact"](key, spec, probe, 8, "t", "v")
    seen.clear()
    fits = []
    real_fit = ns["fit_probe"]
    ns["fit_probe"] = lambda *a, **k: fits.append(1) or real_fit(*a, **k)
    replacements = {} if not enabled else {"if RUN_RESOLUTION_STRESS:": "if True:"}
    run("23084f05", ns, replacements=replacements)
    return seen, fits


def test_enabled_resolution_stress_produces_every_condition_with_the_saved_probe(ns, tmp_path):
    seen, fits = stress_namespace(ns, tmp_path, enabled=True)
    rows = ns["resolution_stress_rows"]
    assert {(r["model_key"], r["condition"]) for r in rows} == {
        (k, c) for k in ("resnet50", "vit") for c in ("original", "96px", "160px")
    }
    assert all(r["n_images"] == 24 for r in rows)
    for key in ("resnet50", "vit"):
        manifest = json.loads((ns["OUT_ROOT"] / "probes" / key / "manifest.json").read_text())
        assert {r["probe_sha256"] for r in rows if r["model_key"] == key} == {
            manifest["probe_file"]["sha256"]
        }
    assert fits == []  # nothing is refitted or reselected
    # Each model sees the 24 originals (longer side 200-206 px), then the photos reduced to 96 and 160 px.
    longest = [max(size) for size in seen]
    assert sum(side >= 200 for side in longest) == 2 * 24
    assert longest.count(96) == 2 * 24 and longest.count(160) == 2 * 24


def test_disabled_resolution_stress_says_it_was_skipped(ns, tmp_path, capsys):
    stress_namespace(ns, tmp_path, enabled=False)
    assert ns["resolution_stress_rows"] == []
    assert "skipped" in capsys.readouterr().out


# --- M4: the activity is validation-only


class Forbidden:
    def __getitem__(self, key):
        raise AssertionError("the validation-only activity read the test split")

    def __getattr__(self, name):
        raise AssertionError("the validation-only activity read the test split")


class DevelopmentSplits(dict):
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("the validation-only activity read the test split")
        return super().__getitem__(key)


def test_activity_never_reads_test_data_and_saves_its_plan_first(ns, tmp_path, capsys):
    rng = np.random.default_rng(1)
    train = [{"id": f"train-{i:03d}", "source_id": f"s{i:03d}", "label_id": i % 6} for i in range(108)]
    ns["splits"] = DevelopmentSplits(train=train)
    ns["ALL_RESULTS"] = {k: {"model": ns["MODEL_REGISTRY"][k]["display"]} for k in ("resnet50", "vit")}
    ns["ALL_FEATURES"] = {
        k: {
            "train": {
                "features": rng.normal(size=(108, 8)).astype(np.float32) + np.arange(108)[:, None] % 6,
                "labels": np.arange(108) % 6,
            },
            "validation": {
                "features": rng.normal(size=(24, 8)).astype(np.float32) + np.arange(24)[:, None] % 6,
                "labels": np.arange(24) % 6,
            },
            "test": Forbidden(),
        }
        for k in ("resnet50", "vit")
    }
    run("f3c4811d", ns, replacements={"if RUN_DATA_EFFICIENCY:": "if False:"})
    run(
        "activity-validation",
        ns,
        replacements={
            "RUN_VALIDATION_ACTIVITY = False  # @param": "RUN_VALIDATION_ACTIVITY = True  # @param",
        },
    )
    activity = ns["OUT_ROOT"] / "activity"
    assert (activity / "plan.json").is_file()
    rows = ns["activity_rows"]
    assert {(r["model_key"], r["train_images"]) for r in rows} == {
        (k, n) for k in ("resnet50", "vit") for n in (36, 72, 108)
    }
    assert all("test" not in field for r in rows for field in r)
    assert all(r["log_loss_change_vs_full"] == 0 for r in rows if r["train_images"] == 108)
    assert "no test data was read" in capsys.readouterr().out


def test_activity_is_off_by_default_and_the_learning_curve_is_labelled_predeclared(ns, capsys):
    run("activity-validation", ns, cut="def validation_activity")
    assert ns["RUN_VALIDATION_ACTIVITY"] is False
    assert "predeclared test curve" in CELLS["4f4233dc"]
    assert "never reads the test split" in CELLS["4f4233dc"]


# --- M5: the diagnostic views are shown


def diagnostics_namespace(ns, all_correct):
    keys = list(ns["MODEL_REGISTRY"])
    test = [
        {
            "id": f"test-{i:03d}",
            "source_id": f"s{i}",
            "label": ns["CLASS_KEYS"][i % 6],
            "label_id": i % 6,
            "image": colour_image(i % 6, i),
        }
        for i in range(12)
    ]
    ns["splits"] = {"test": test}
    results = {}
    for m, key in enumerate(keys):
        truth = np.arange(12) % 6
        pred = truth.copy() if all_correct else np.where(np.arange(12) < m, (truth + 1) % 6, truth)
        probs = np.full((12, 6), 0.02)
        probs[np.arange(12), pred] = 0.9
        metrics = ns["classification_metrics"](truth, probs)
        results[key] = {
            "model": ns["MODEL_REGISTRY"][key]["display"],
            "test_probs": probs.astype(np.float32),
            "test_pred": metrics["predicted_ids"],
            "probe_metrics": metrics,
        }
    ns["ALL_RESULTS"] = results
    run("a60da6ff", ns)
    return ns


def test_confusion_recall_and_a_real_disagreement_gallery_are_shown(ns, capsys):
    diagnostics_namespace(ns, all_correct=False)
    out = capsys.readouterr().out
    assert "Recall by species" in out
    assert ns["plt"].shown == 2  # confusion matrices, then the gallery
    ids = [row["image_id"] for row in ns["gallery"]]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert ns["gallery"][0]["difficulty_category"] == "shared_hard_case"
    assert all(len(line) <= 40 for text in ns["plt"].texts for line in text.splitlines())
    row = ns["prediction_rows"][0]
    assert {f"score_{label}" for label in ns["CLASS_KEYS"]} <= set(row)


def test_no_mistakes_is_said_rather_than_invented(ns, capsys):
    diagnostics_namespace(ns, all_correct=True)
    assert "there is no mistake or disagreement to show" in capsys.readouterr().out
    assert ns["plt"].shown == 1


def test_pca_and_training_examples_are_displayed(ns):
    rng = np.random.default_rng(2)
    ns["ALL_RESULTS"] = {k: {"model": k} for k in ("resnet50", "vit")}
    ns["ALL_FEATURES"] = {
        k: {
            "train": {"features": rng.normal(size=(30, 8))},
            "test": {
                "features": rng.normal(size=(12, 8)),
                "labels": np.arange(12) % 6,
                "source_ids": np.array([f"s{i}" for i in range(12)]),
            },
        }
        for k in ("resnet50", "vit")
    }
    ns["OUT_ROOT"].joinpath("pca").mkdir(parents=True, exist_ok=True)
    run("cd6f5ec4", ns)
    assert ns["plt"].shown == 2
    assert "plt.close(fig)" not in CELLS["cd6f5ec4"]
    ns["splits"] = {
        "train": [
            {"label": label, "image": colour_image(i, i), "common_name": label, "scientific_name": label}
            for i, label in enumerate(ns["CLASS_KEYS"])
        ]
    }
    run("gallery-train", ns)
    assert ns["plt"].shown == 3


def test_metrics_are_explained_at_first_use():
    guide = CELLS["755a7c8d"]
    for term in ("Accuracy", "Macro F1", "Top-3", "Log-loss", "Worked contrast"):
        assert term in guide


# --- M2 and BYOD minors: staged, contained, preflighted


def byod_fixture(root, labels=("001", "NA"), subdir=""):
    rows = ["filename,label,split"]
    for c, label in enumerate(labels):
        for i in range(30):
            split = "train" if i < 20 else "validation" if i < 25 else "test"
            name = f"{subdir}c{c}_{i:02d}.png"
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (c * 90, i * 7, 255 - i * 5)).save(path)
            rows.append(f"{name},{label},{split}")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def zip_dir(directory, target, extra=None):
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return target


@pytest.fixture
def byod(ns):
    run("71cf319a", ns)
    return ns


def summary(out):
    return sorted(
        (r["split"], r["source_id"], r["label"], r["pixel_sha256"]) for part in out.values() for r in part
    )


def test_directory_and_zip_inputs_are_equivalent_with_nested_folders(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data", subdir="images/")
    from_dir, labels = byod["load_byod_explicit"](directory)
    from_zip, zip_labels = byod["load_byod_explicit"](zip_dir(directory, tmp_path / "data.zip"))
    assert labels == zip_labels == ["001", "NA"]
    assert summary(from_dir) == summary(from_zip)
    assert {k: len(v) for k, v in from_zip.items()} == {"train": 40, "validation": 10, "test": 10}


def test_a_same_named_unreferenced_member_cannot_replace_a_referenced_image(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data")
    decoy = Image.new("RGB", (16, 16), (1, 2, 3))
    buffer = io.BytesIO()
    decoy.save(buffer, format="PNG")
    archive = zip_dir(directory, tmp_path / "data.zip", extra={"unreferenced/c0_00.png": buffer.getvalue()})
    out, _ = byod["load_byod_explicit"](archive)
    direct, _ = byod["load_byod_explicit"](directory)
    assert summary(out) == summary(direct)


def test_colliding_zip_members_are_refused(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data")
    archive = zip_dir(directory, tmp_path / "data.zip", extra={"C0_00.PNG": b"x"})
    with pytest.raises(ValueError, match="same file"):
        byod["load_byod_explicit"](archive)


def test_an_incomplete_retry_never_reuses_an_earlier_upload(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data")
    byod["load_byod_explicit"](zip_dir(directory, tmp_path / "complete.zip"))
    (directory / "c1_03.png").unlink()
    with pytest.raises(ValueError, match="missing image"):
        byod["load_byod_explicit"](zip_dir(directory, tmp_path / "incomplete.zip"))
    staging = tmp_path / "byod_modern_image"
    assert (
        len([p for p in staging.iterdir()]) == 1
    )  # the failed attempt's folder was removed, the earlier one kept


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("../outside.png", "Unsafe filename"),
        ("/etc/outside.png", "Unsafe filename"),
        ("C:/outside.png", "Unsafe filename"),
        ("sub/../../outside.png", "Unsafe filename"),
    ],
)
def test_csv_paths_must_stay_inside_the_dataset(byod, tmp_path, filename, message):
    directory = byod_fixture(tmp_path / "data")
    Image.new("RGB", (16, 16), (9, 9, 9)).save(tmp_path / "outside.png")
    table = directory / "labels.csv"
    lines = table.read_text().splitlines()
    lines[1] = f"{filename},001,train"
    table.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=message):
        byod["load_byod_explicit"](directory)
    assert directory.is_dir()  # a user directory is never removed


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_symbolic_links_in_a_directory_are_refused(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data")
    target = tmp_path / "elsewhere.png"
    Image.new("RGB", (16, 16), (9, 9, 9)).save(target)
    (directory / "c0_00.png").unlink()
    (directory / "c0_00.png").symlink_to(target)
    with pytest.raises(ValueError, match="Symbolic links"):
        byod["load_byod_explicit"](directory)


def test_cheap_requirements_are_checked_before_any_image_is_decoded(byod, tmp_path, monkeypatch):
    directory = byod_fixture(tmp_path / "data", labels=("001", "NA", "other"))
    table = directory / "labels.csv"
    lines = [
        line for line in table.read_text().splitlines() if not line.endswith(",test") or "c1_" not in line
    ]
    table.write_text("\n".join(lines) + "\n")
    decoded = []

    def refuse(*args, **kwargs):
        decoded.append(args)
        raise AssertionError("an image was decoded before the table checks")

    monkeypatch.setattr(byod["Image"], "open", refuse)
    with pytest.raises(ValueError, match="insufficient explicit split counts"):
        byod["load_byod_explicit"](directory)
    assert decoded == []


def test_byod_run_writes_predictions_class_metrics_and_an_inventory(byod, tmp_path):
    directory = byod_fixture(tmp_path / "data", labels=("001", "NA", "long label " + "é" * 40))
    seen = []
    transform = fake_transform(seen)

    class Model(FakeBackbone):
        head_hidden_size = 8
        num_features = 8

    byod["load_backbone"] = lambda key: (Model(), transform, {}, None, 0.0, 0.0)
    byod["BYOD_PATH"] = str(directory)
    byod["BYOD_MODEL_KEYS"] = ["resnet50", "vit"]
    run("71cf319a", byod, replacements={"if USE_BYOD:": "if True:"})
    root = byod["OUT_ROOT"] / "byod"
    for name in ("summary.csv", "predictions.csv", "class_metrics.csv", "provenance.json"):
        assert (root / name).is_file()
    header = (root / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "score_001" in header and "score_NA" in header and "knn_predicted_label" in header
    manifest = json.loads((root / "probes/vit/manifest.json").read_text())
    assert manifest["class_order"] == ["001", "NA", "long label " + "é" * 40]


# --- minors and revision identity


def test_privacy_requirements_and_runtime_guidance_are_stated():
    byod_text = CELLS["766150f9"]
    assert "Enforced requirements" in byod_text and "Recommended limits" not in byod_text
    assert "**Privacy.**" in byod_text and "not an on-premises system" in byod_text
    assert "Python 3.12 `timm` runtime" not in CELLS["6fd0e1d2"]
    assert "Use the default tier" not in CELLS["guided-04"]
    assert "not full image-to-label latency" in CELLS["6e025182"]


def test_notebook_revision_is_recorded():
    # 0.3.0 moved the notebook to the uv isolated environment; the 0.2.0 review fixes above still hold.
    assert NB["metadata"]["dimer"]["workshop_revision"] == "0.3.0-candidate"
    assert 'NOTEBOOK_REVISION = "0.3.0-candidate"' in CELLS["4f54b044"]


def test_committed_notebook_is_clean():
    for cell in NB["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
