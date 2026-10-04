"""The modern image workshop runs in a uv isolated environment, not in the notebook kernel (revision 0.3.0).

Static checks of the setup cells, the carried hash lock and the cell line limit, plus a POSIX-only run of the
cell-routing worker with the test interpreter standing in for the isolated one. None of this is a hosted run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Modern_Image_Classification_Workshop.ipynb"
LOCK = ROOT / "tools" / "modern-image-workshop-requirements.lock"
REQUIREMENTS_IN = ROOT / "tools" / "modern-image-workshop-requirements.in"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE = [(c["id"], "".join(c["source"])) for c in NB["cells"] if c["cell_type"] == "code"]
CELLS = {c["id"]: "".join(c["source"]) for c in NB["cells"]}
KERNEL_MARK = "# dimer: kernel cell"

sys.path.insert(0, str(ROOT / "tools"))
import build_modern_workshop_runtime as runtime_tool  # noqa: E402


def assigned_literal(source: str, name: str):
    """The value of a top-level ``name = <literal>`` (or ``name = json.loads(<literal>)``) in a cell."""
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            value = node.value
            if isinstance(value, ast.Call):
                (value,) = value.args
            return ast.literal_eval(value)
    raise AssertionError(f"{name} is not assigned in the cell")


def test_no_kernel_install_and_no_restart_guard():
    for cell_id, source in CODE:
        assert not re.search(r"pip\s+install", source), cell_id
        assert '"-m", "pip"' not in source and "'-m', 'pip'" not in source, cell_id
        for marker in ("NUMPY_PRELOADED", "if stale:", "Restart session", "check_call"):
            assert marker not in source, (cell_id, marker)
    markdown = "\n".join(s for c, s in CELLS.items() if c not in dict(CODE))
    assert "Restart session** and then **Run all**" not in markdown
    assert "stale-module" not in markdown


def test_only_the_two_setup_cells_run_in_the_kernel_and_they_come_first():
    kernel = [cell_id for cell_id, source in CODE if KERNEL_MARK in source]
    assert kernel == ["uv-install", "uv-router"]
    assert [cell_id for cell_id, _ in CODE][:2] == kernel


def test_setup_cell_builds_a_managed_python_and_installs_the_lock_with_hashes_only():
    cell = CELLS["uv-install"]
    assert assigned_literal(cell, "MANAGED_PYTHON") == "3.12.12"
    assert '"--managed-python", "--python", MANAGED_PYTHON' in cell
    assert '"--require-hashes", "--only-binary", ":all:"' in cell
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in cell
    assert "hashlib.sha256(wheel).hexdigest() != UV_SHA256" in cell
    assert "hashlib.sha256(LOCK_TEXT.encode(\"utf-8\")).hexdigest() != LOCK_SHA256" in cell


def test_carried_lock_is_the_repository_lock_and_every_package_is_hashed():
    cell = CELLS["uv-install"]
    lock_text = LOCK.read_text(encoding="utf-8")
    assert assigned_literal(cell, "LOCK_TEXT") == lock_text
    digest = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert assigned_literal(cell, "LOCK_SHA256") == digest
    assert NB["metadata"]["dimer"]["carried_files"]["requirements.lock.txt"]["sha256"] == digest
    entries = re.split(r"\n(?=[A-Za-z0-9][A-Za-z0-9._-]*==)", lock_text.split("\n", 2)[2])
    assert len(entries) == assigned_literal(cell, "LOCKED_PACKAGES") == 53
    for entry in entries:
        assert "--hash=sha256:" in entry, entry.splitlines()[0]
    assert "--generate-hashes" in lock_text and "x86_64-manylinux_2_28" in lock_text


def test_direct_pins_are_unchanged_and_agree_across_input_lock_and_runtime_cell():
    expected = {
        "torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0", "timm": "1.0.29",
        "safetensors": "0.8.0", "numpy": "2.5.3", "pillow": "11.3.0", "huggingface-hub": "0.36.2",
        "matplotlib": "3.10.6",
    }
    assert assigned_literal(CELLS["aa91b536"], "PINS") == expected
    pins_in = dict(
        line.split("==") for line in REQUIREMENTS_IN.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert pins_in == expected
    locked = dict(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", LOCK.read_text(encoding="utf-8"), re.M))
    assert {name: locked[name] for name in expected} == expected


def test_runtime_cell_checks_the_pins_without_installing():
    cell = CELLS["aa91b536"]
    assert "mismatched" in cell and "raise RuntimeError" in cell
    assert "subprocess" not in cell


def test_router_starts_one_worker_on_the_isolated_interpreter():
    cell = CELLS["uv-router"]
    assert "IsolatedRuntime(ISOLATED_PYTHON)" in cell
    assert "[str(python), \"-c\", _WORKER_SOURCE" in cell
    assert 'MPLBACKEND="Agg"' in cell
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN"):
        assert f'"{name}"' in cell
    assert "input_transformers_cleanup.append(_route_to_isolated_runtime)" in cell


def test_setup_cell_is_current_with_the_lock():
    assert runtime_tool.main(["--check"]) == 0


def test_no_cell_line_exceeds_2000_characters():
    longest = max((len(line), cid) for cid, text in CELLS.items() for line in text.split("\n"))
    assert longest[0] <= 2000, longest


@pytest.mark.parametrize(
    ("cell_id", "name", "digest"),
    [
        ("e04f3109", "SAMPLE_ROWS", "57b663ea54af200dda42847c2f0a9df78af8a7f97aa896755ddfed5aa38547c2"),
        ("f470276e", "MANIFESTS", "8f7722ae83f8fde4bf4621fe7481a3bebcec3f772fd3009d3f57d122b29bcea1"),
    ],
)
def test_split_literals_equal_the_original_single_line_literals(cell_id, name, digest):
    # The digests are of the r'''...''' literals in revision 0.2.0 (blob fd59129e).
    source = CELLS[cell_id]
    text = assigned_literal(source, name)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
    call = next(
        n.value for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and n.targets[0].id == name
    )
    pieces = [ast.get_source_segment(source, call.args[0])]
    assert all(len(line.strip()) <= 1010 for line in pieces[0].splitlines())
    assert ast.literal_eval(call.args[0]) == text
    value = json.loads(text)
    assert len(value) == (180 if name == "SAMPLE_ROWS" else 6)


def router_namespace():
    """The router cell's worker source and IsolatedRuntime class, without IPython or the input transformer."""
    tree = ast.parse(CELLS["uv-router"])
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "IPython":
            continue
        worker_source = isinstance(node, ast.Assign) and node.targets[0].id == "_WORKER_SOURCE"
        if worker_source or isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            keep.append(node)
    namespace = {"os": os, "sys": sys, "subprocess": __import__("subprocess")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "uv-router", "exec"), namespace)
    return namespace


@pytest.mark.skipif(os.name == "nt", reason="the worker passes pipe descriptors (POSIX only)")
def test_routed_cells_share_one_persistent_namespace_and_errors_stop_the_run(capsys):
    ns = router_namespace()
    shown = []
    worker = ns["IsolatedRuntime"](sys.executable, display=lambda bundle, raw=True: shown.append(bundle))
    try:
        worker.run("import os\nvalue = 21\nprint('first', value)")
        worker.run("print('second', value * 2, os.environ.get('MPLBACKEND'), os.environ.get('PYTHONPATH'))")
        worker.run("value * 3")
        with pytest.raises(ns["IsolatedCellError"], match="boom"):
            worker.run("raise ValueError('boom')")
        worker.run("print('after error', value)")
    finally:
        worker.close()
    out = capsys.readouterr().out
    assert "first 21" in out and "second 42 Agg None" in out and "after error 21" in out
    assert shown and shown[0]["text/plain"] == "63"
