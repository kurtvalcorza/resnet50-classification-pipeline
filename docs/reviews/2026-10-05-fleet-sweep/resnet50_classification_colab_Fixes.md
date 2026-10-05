# Fleet-sweep fixes: `resnet50_classification_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `86c899d`. All changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`); the notebook is regenerated. Status and release labels are unchanged. The workshop notebook
(`DIMER_Modern_Image_Classification_Workshop.ipynb`, its own generator, already isolated) is untouched;
`tools/build_modern_workshop_runtime.py --check` passes.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → the fleet's shared `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (45 packages compiled from the unchanged pyproject pins, `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–9), What to notice + Check your reasoning after each, Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. The only recorded run (Kaggle T4, 2026-09-14) logs a pass and no per-stage numbers, so the worked answers are qualitative and quote no values. | opening, Sections 4–9 markdown, closing | `test_swp_g_guided_layer_is_present_and_infrastructure_is_collapsed`, `test_swp_g_checkpoint_answers_are_qualitative_because_no_numbers_are_recorded` |
| SWP-A (quality asserts) | Not flagged | No quality assert in the cell source. | — | — |
| SWP-F (frozen re-run) | Not present | `pipe.fit` returns a separate fine-tuned pipeline; `pipe` is not trained in place. | — | — |
| SWP-B (BYOD upload only) | Fixed | Confirmed: both BYOD branches used only `files.upload()`. Added `BYOD_IMAGE_PATH` (Section 4) and `BYOD_DATASET_PATH` (Section 8) so Kaggle/Jupyter work; the upload is a guarded fallback (off Colab, cancelled or multi-file upload, an undecodable image each give a message naming the file or rule). | Sections 4, 8 | `test_swp_b_image_path_works_without_colab`, `test_swp_b_dataset_path_works_and_cancelled_upload_is_refused` |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- New `BYOD_IMAGE_PATH` and `BYOD_DATASET_PATH` fields.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; both BYOD blocks run with stand-in files. Plumbing evidence, not model evidence.
- `pytest` with CI's pins (installing `timm` pulled the PyPI torch build, so the torch-backed tests ran too): 90 passed before → 97 passed after.
- `build_notebook.py --check` up to date; `build_modern_workshop_runtime.py --check` OK; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab runtime (no restart expected), then a re-run of the Section 10 export cell; record per-stage numbers so later checkpoint answers can quote them.
- The REL12 BYOD run (`USE_BYOD_DATASET = True` with `BYOD_DATASET_PATH`).
- A full Notebook Review Framework v1 review has not been done.
