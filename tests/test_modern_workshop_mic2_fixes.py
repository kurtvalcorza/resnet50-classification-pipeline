"""Checks for the 2026-10-12 review findings MIC2-M1 and MIC2-m1..m4 on the modern image workshop.

The tests execute the notebook's own code cells with inert stand-ins (no weights, photos or network). They are
logic and source checks, not model runs or Colab evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import json
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

LARGE_LABEL_SET_MODELS = ("convnext", "vit", "eva02")
IN1K_ONLY_MODELS = ("resnet50", "mobilenetv4", "swinv2")


def run(cell_id, ns, cut=None):
    text = CELLS[cell_id]
    if cut:
        text = text[: text.index(cut)]
    exec(compile(text, cell_id, "exec"), ns)
    return ns


def definitions(cell_id, ns):
    """Execute only the function definitions of a cell."""
    tree = ast.parse(CELLS[cell_id])
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(module, cell_id, "exec"), ns)
    return ns


@pytest.fixture
def ns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from safetensors.torch import load_file, save_file

    namespace = {"display": print, "np": np, "torch": torch, "F": torch.nn.functional, "Image": Image,
                 "load_file": load_file, "save_file": save_file}
    run("4f54b044", namespace)
    run("e04f3109", namespace, cut="records = fetch_dataset()")
    run("f470276e", namespace)
    run("f370927c", namespace)
    run("6e025182", namespace, cut="ALL_RESULTS={}")
    run("22e6f1e2", namespace, cut="model_metric_rows=[]")
    namespace["RUNTIME"] = {"python": "test"}
    namespace["DEVICE"] = "cpu"
    return namespace


# --- MIC2-M1: pretraining label sets are disclosed per checkpoint


def test_each_checkpoint_declares_its_pretraining_labels(ns):
    registry = ns["MODEL_REGISTRY"]
    for key in LARGE_LABEL_SET_MODELS:
        assert registry[key]["target_species_in_labels"] == "all six"
        assert any(tag in registry[key]["pretraining_labels"] for tag in ("12k", "21k", "22k"))
    for key in IN1K_ONLY_MODELS:
        assert registry[key]["pretraining_labels"] == "ImageNet-1k"
        assert registry[key]["target_species_in_labels"].startswith("house finch only")
    provenance = CELLS["22e6f1e2"]
    assert '"pretraining_labels":spec["pretraining_labels"]' in provenance
    assert '"target_species_in_pretraining_labels":spec["target_species_in_labels"]' in provenance
    assert '"pretraining_labels":MODEL_REGISTRY[key]["pretraining_labels"]' in CELLS["8e6fccd6"]


def test_label_set_overlap_is_disclosed_and_not_credited_to_architecture():
    section2 = CELLS["a395b794"]
    assert "Architectural progression" not in section2
    for wnid in ("n01532325", "n01532829", "n01534582", "n01535469", "n01536035", "n01536644"):
        assert wnid in section2
    assert "| ConvNeXt-Tiny | `in12k_ft_in1k` |" in section2 and "| all six |" in section2
    assert "masked image modeling on ImageNet-22k images (no labels)" in section2
    assert "consistent with label-set exposure confounding" in section2
    # Only house finch is an exact ImageNet-1k class; dark-eyed junco is covered by the broader "junco" class.
    assert "house finch exactly and a broader" in section2
    for cell_id in ("396e7ade", "c34ab144", "guided-02"):
        assert "confounding the comparison" in CELLS[cell_id]
    for stale in ("outside a clean shared six-class mapping", "make feature transfer non-trivial",
                  "Which checkpoint produced the strongest frozen representation?",
                  "justify its compute cost"):
        assert stale not in MARKDOWN
    assert "not why" in CELLS["c33a2d79"]


# --- MIC2-m1 and MIC2-m2: fixed recipe, cap flag and intervals are shown


def test_fixed_recipe_and_cap_are_explained():
    assert "It is not tuned per backbone." in CELLS["8b54b6df"]
    assert "1,000-epoch cap" in CELLS["8b54b6df"]
    assert "One fixed probe recipe, not tuned per backbone" in CELLS["c34ab144"]
    assert "one nested-subset illustration" in CELLS["4f4233dc"]
    # Peer-review follow-up: diagnostic wording only, no causal claims about the cap or the curve.
    for stale in ("still improving", "still falling", "barely trained", "real probe", "sampling noise",
                  "cannot separate", "Overlapping intervals", "identical six-class linear probe",
                  "exact same"):
        assert all(stale not in CELLS[c["id"]] for c in NB["cells"] if c["id"] != "uv-install"), stale
    assert "extending training might change the selection" in CELLS["8b54b6df"]
    assert "validation-selection diagnostic" in CELLS["guided-01"]
    assert "Validation-selection diagnostic" in CELLS["activity-validation"]
    assert "synthetic source-detail degradation" in CELLS["fb5d8f70"]
    assert '"preprocessing":ALL_RESULTS[key]["preprocessing"]' in CELLS["22e6f1e2"]
    assert "**Softmax score**" in CELLS["guided-05"] and "**Epoch selection**" in CELLS["guided-05"]


@pytest.mark.parametrize(("b", "c", "expected"), [(0, 0, 1.0), (5, 0, 0.0625), (3, 3, 1.0), (10, 2, 0.0386)])
def test_exact_mcnemar_matches_the_binomial_closed_form(b, c, expected):
    helpers = definitions("7aaa53fc", {"np": np, "math": __import__("math")})
    assert helpers["exact_mcnemar_p"](b, c) == pytest.approx(expected, abs=1e-4)


def test_paired_table_counts_discordant_photos_against_the_top_probe(capsys):
    run("7aaa53fc", fake_results())
    out = capsys.readouterr().out
    # eva02 (43 right) is top. resnet50 is right on photos 0-23 and eva02 on 0-42: 19 only-top, 0 only-other.
    row = next(line for line in out.splitlines() if line.startswith("resnet50") and "0.0000" in line)
    assert row.split()[1:3] == ["19", "0"]
    assert "Bonferroni" in out and "illustrative" in out
    assert "cannot separate" not in out


def test_observations_crossing_splits_stop_the_split_cell(ns):
    rows = [(f"r{i}", ns["CLASS_KEYS"][i % 6], i, i // 2, "o", 1, "x", "jpg") for i in range(180)]
    records = [{"source_id": rid, "label": lab, "label_id": ns["CLASS_TO_ID"][lab], "photo_id": pid,
                "observation_id": oid, "observer": obs, "bytes": b, "sha256": d, "pixel_sha256": str(pid)}
               for rid, lab, pid, oid, obs, b, d, _ext in rows]
    ns["records"] = records
    with pytest.raises(RuntimeError, match="observations have photos in more than one split"):
        run("1b442017", ns)
    for i, r in enumerate(records):
        r["observation_id"] = i
    run("1b442017", ns)
    assert ns["OBSERVATIONS_CROSSING_SPLITS"] == 0


def test_wilson_interval_matches_the_closed_form():
    helpers = definitions("7aaa53fc", {"np": np})
    wilson = helpers["wilson_interval"]
    lo, hi = wilson(24, 48)
    assert lo == pytest.approx(0.36389, abs=1e-4) and hi == pytest.approx(0.63611, abs=1e-4)
    assert wilson(0, 48)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson(48, 48)[1] == pytest.approx(1.0, abs=1e-12)
    assert wilson(43, 48)[0] < 43 / 48 < wilson(43, 48)[1]
    for bad in ((49, 48), (-1, 48), (0, 0)):
        with pytest.raises(ValueError):
            wilson(*bad)


def fake_results(accuracy_override=None):
    labels = np.repeat(np.arange(6), 8)
    results, features, registry = {}, {}, {}
    for i, (key, correct, cap) in enumerate((("resnet50", 24, False), ("eva02", 43, True))):
        pred = labels.copy()
        pred[correct:] = (pred[correct:] + 1) % 6
        acc = correct / 48 if accuracy_override is None else accuracy_override
        registry[key] = {"pretraining_labels": f"labels-{i}"}
        features[key] = {"test": {"labels": labels}}
        results[key] = {
            "model": key, "feature_dim": 8, "knn_metrics": {"accuracy": 0.5},
            "probe_metrics": {"accuracy": acc, "macro_f1": 0.5, "top3_accuracy": 0.9, "log_loss": 1.0},
            "test_pred": pred, "best_epoch": 1000 if cap else 35, "selected_at_epoch_cap": cap,
        }
    return {"np": np, "MODEL_REGISTRY": registry, "ALL_RESULTS": results, "ALL_FEATURES": features,
            "PROBE_EPOCHS": 1000}


def test_section_11_prints_counts_intervals_epochs_and_cap_flags(capsys):
    ns = fake_results()
    run("7aaa53fc", ns)
    out = capsys.readouterr().out
    assert "95% interval" in out and "At cap" in out
    resnet = next(line for line in out.splitlines() if line.startswith("resnet50") and "24/48" in line)
    assert "[0.364, 0.636]" in resnet and resnet.rstrip().endswith("no")
    eva = next(line for line in out.splitlines() if line.startswith("eva02") and "43/48" in line)
    assert " 1000 " in eva and eva.rstrip().endswith("yes")


def test_interval_display_refuses_a_count_that_disagrees_with_the_metric():
    with pytest.raises(RuntimeError, match="disagrees"):
        run("7aaa53fc", fake_results(accuracy_override=0.99))


def test_terminal_summary_prints_the_intervals():
    summary = CELLS["3c8af07b"]
    assert "probe_correct(key)" in summary and "wilson_interval(correct,n)" in summary
    assert summary.index("wilson_interval") < summary.index('print(f"Outputs: {OUT_ROOT}/")')


# --- MIC2-m3: BYOD refusals are one message, integrity failures still stop (DAT20)


def dataset(root, classes=2, corrupt=False):
    root.mkdir(parents=True)
    rows = ["filename,label,split"]
    for c in range(classes):
        for i in range(30):
            split = "train" if i < 20 else "validation" if i < 25 else "test"
            name = f"c{c}_{i:02d}.png"
            Image.new("RGB", (16, 16), (c * 90, i * 7, 255 - i * 5)).save(root / name)
            rows.append(f"{name},L{c},{split}")
    if corrupt:
        (root / "c0_00.png").write_bytes(b"not an image")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("empty", "BYOD_PATH is empty"),
        ("no_labels", "requires labels.csv"),
        ("one_class", "BYOD must contain 2..20 classes"),
    ],
)
def test_a_refused_byod_input_prints_one_message_and_clears_stale_outputs(
    ns, tmp_path, capsys, case, expected
):
    paths = {"empty": "", "no_labels": str(tmp_path / "empty_dir"), "one_class": None}
    (tmp_path / "empty_dir").mkdir()
    if case == "one_class":
        paths["one_class"] = str(dataset(tmp_path / "one", classes=1))
    stale = ns["OUT_ROOT"] / "byod"
    stale.mkdir(parents=True)
    (stale / "summary.csv").write_text("from an earlier upload\n", encoding="utf-8")
    ns.update(USE_BYOD=True, BYOD_PATH=paths[case])
    run("71cf319a", ns)  # no exception escapes, so no traceback is shown
    out = capsys.readouterr().out
    assert out.count("BYOD refused") == 1 and expected in out
    assert "run this cell again" in out
    assert not stale.exists()  # nothing is left where it could be read as this attempt's result
    attempts = ns["OUT_ROOT"] / "byod_attempts"
    status = json.loads(next(attempts.glob("*-refused/status.json")).read_text(encoding="utf-8"))
    assert status["status"] == "refused" and expected in status["reason"]
    kept = next(attempts.glob("*-superseded-byod")) / "summary.csv"
    assert kept.read_text(encoding="utf-8") == "from an earlier upload\n"  # earlier results are not deleted
    assert ns["byod_splits"] is None and ns["byod_summary"] == []


def test_an_undecodable_image_still_stops_with_an_error(ns, tmp_path):
    ns.update(USE_BYOD=True, BYOD_PATH=str(dataset(tmp_path / "bad", corrupt=True)))
    with pytest.raises(OSError):
        run("71cf319a", ns)


def test_byod_disabled_path_is_unchanged(ns, capsys):
    ns.update(USE_BYOD=False)
    run("71cf319a", ns)
    assert capsys.readouterr().out.strip() == "BYOD disabled on the canonical Run all path."


# --- MIC2-m4: spec 2.5, notebook_kind and notebook_profile


def test_workshop_declares_spec_kind_and_profile():
    dimer = NB["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5"
    assert dimer["notebook_kind"] == "WORKSHOP" and dimer["notebook_profile"] == "E2E"
    assert dimer["notebook_mode"] == "WORKSHOP"
    assert '"notebook_spec":"2.5","notebook_kind":"WORKSHOP","notebook_profile":"E2E"' in CELLS["22e6f1e2"]


def test_validator_rejects_a_stale_workshop_declaration(monkeypatch):
    path = ROOT / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_workshop_declaration()
    stale = NOTEBOOK.read_text(encoding="utf-8").replace('"notebook_spec": "2.5"', '"notebook_spec": "2.1"')
    monkeypatch.setattr(module, "_read", lambda path: stale)
    with pytest.raises(module.ValidationError, match="notebook_spec"):
        module.validate_workshop_declaration()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("**Pedagogical mode:** `WORKSHOP`", "**Pedagogical mode:** `GUIDED`", "Pedagogical mode"),
        ("**Notebook profile:** `E2E`", "**Notebook profile:** `SMOKE`", "Notebook profile"),
        ('\\"pedagogical_mode\\":\\"WORKSHOP\\"', '\\"pedagogical_mode\\":\\"GUIDED\\"', "pedagogical_mode"),
        ('\\"notebook_profile\\":\\"E2E\\"', '\\"notebook_profile\\":\\"SMOKE\\"', "notebook_profile"),
    ],
)
def test_validator_checks_every_field_in_the_opening_cell_and_provenance(monkeypatch, old, new, message):
    path = ROOT / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    monkeypatch.setattr(module, "_read", lambda path: text.replace(old, new))
    with pytest.raises(module.ValidationError, match=message):
        module.validate_workshop_declaration()
