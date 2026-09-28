"""Execute every code cell of the revised workshop notebook in order on CPU, with real data and checkpoints.

Usage: python rn_preflight.py <notebook> <workdir> <out.json>
Enables resolution stress and the validation-only activity; everything else stays at its defaults.
"""
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
notebook, workdir, out_json = sys.argv[1:4]
cells = [c for c in json.loads(Path(notebook).read_text())["cells"] if c["cell_type"] == "code"]
Path(workdir).mkdir(parents=True, exist_ok=True)
os.chdir(workdir)

ns = {"display": print, "__name__": "__main__"}
log = []
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    src = src.replace('RUN_RESOLUTION_STRESS = False  # @param', 'RUN_RESOLUTION_STRESS = True  # @param')
    src = src.replace('RUN_VALIDATION_ACTIVITY = False  # @param', 'RUN_VALIDATION_ACTIVITY = True  # @param')
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"cell-{cell['id']}", "exec"), ns)
        status = "ok"
    except Exception:
        status = "error"
        buf.write(traceback.format_exc())
    elapsed = time.perf_counter() - t0
    log.append({"cell": cell["id"], "status": status, "seconds": round(elapsed, 1), "output": buf.getvalue()[-6000:]})
    print(f"[{i+1}/{len(cells)}] {cell['id']} {status} {elapsed:.0f}s", flush=True)
    Path(out_json).write_text(json.dumps({"cells": log}, indent=2))
    if status == "error":
        print(buf.getvalue()[-3000:], flush=True)
        break

# 5-NN: the tie-aware decision the notebook now scores versus argmax of the vote fractions it scored before.
import numpy as np  # noqa: E402

knn = {}
if "ALL_FEATURES" in ns:
    for key, feats in ns["ALL_FEATURES"].items():
        _m, _s, (tr, te) = ns["standardize_from_train"](feats["train"]["features"], feats["test"]["features"])
        pred, probs = ns["knn_predict"](tr, feats["train"]["labels"], te, ns["KNN_K"])
        old = probs.argmax(1)
        y = feats["test"]["labels"]
        knn[key] = {
            "tie_aware_accuracy": float(np.mean(pred == y)),
            "argmax_accuracy": float(np.mean(old == y)),
            "rows_with_tied_top_vote": int(sum((p == p.max()).sum() > 1 for p in probs)),
            "rows_whose_decision_changes": int(np.sum(pred != old)),
            "changed_image_ids": [str(feats["test"]["ids"][i]) for i in np.flatnonzero(pred != old)],
        }
summary = {
    "cells": log,
    "knn_tie_comparison": knn,
    "main": {k: {"knn": r["knn_metrics"]["accuracy"], "probe": r["probe_metrics"]["accuracy"],
                 "macro_f1": r["probe_metrics"]["macro_f1"], "log_loss": r["probe_metrics"]["log_loss"],
                 "best_epoch": r["best_epoch"], "reload_diff": r["probe_reload_max_abs_probability_diff"]}
             for k, r in ns.get("ALL_RESULTS", {}).items()},
    "resolution_stress": ns.get("resolution_stress_rows"),
}
Path(out_json).write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps({"knn": knn, "main": summary["main"]}, indent=2, default=str))
