# Release verification

`tutorials/resnet50_classification_colab.ipynb` (`E2E`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, `standalone: true` and `generated_from` (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/resnet50_classification_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ResNet50ClassificationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `predict`, `evaluation_report`), the ceiling print (`NUM_CLASSES`, `MAX_IMAGE_SIDE`, `MAX_BATCH`),
  the exports, the learner-facing classification statements (argmax decision rule, uncalibrated
  softmax, no shipped threshold, rank-ordered scores) and the gated-off BYOD default listed in
  the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `timm.create_model` / `from timm import` / `from torchvision import` /
  `from transformers import` / `from huggingface_hub import` use **outside the carried module cell**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchvision` wheels plus `timm`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - synthetic 256×256 gradient sample generated in code with its RGB SHA-256 printed and the
     ceilings (`NUM_CLASSES` 1000, `MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 64) surfaced;
   - pinned `timm/resnet50.a1_in1k` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/resnet50-a1/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - classification through `predict(image, top_k=5)` with `decision_rule == 'argmax'` and a
     rank-ordered top-5 list;
   - `validate_inputs` writes `outputs/resnet50_classification_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the oversized probe);
   - `evaluation_report` writes `outputs/resnet50_classification_evaluation_report.json` with verdict `not-measurable`
     on the synthetic sample (no ground truth), stated as such;
   - `outputs/resnet50_classification_result.json` and `outputs/resnet50_classification_top_k.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/resnet50_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/resnet50_classification_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `b7e787d` / `77dc7ebd3d7b` | Kaggle T4 (`kurtvalcorza/dimer-nb2-resnet50-classification` v1) | Default sample path | 182.3 s | **PASSED** — 10/10 ok code cells executed cleanly, 8 files, 103 MB staged |

## Current status

No clean-runtime execution of the notebook has been recorded yet; clean GPU execution evidence is now recorded below. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Two facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of all three manifest entries into a fresh `weights/resnet50-a1/` has not been
executed), and the standalone carrier itself — executing the carried module cell in a runtime that has no
repository checkout — has been validated statically only (parity PASS), never run; the clean run will be the
first execution of the standalone path, of the staging path, and of the CPU inference path against the real weights.

## Supplemental modern image classification workshop — `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb`

This entry applies only to the supplemental workshop notebook, not the primary tutorial executions above.

### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Modern_Image_Classification_Workshop.ipynb) and authorized merging PR #7 (merge commit `0546841`). The file is archived byte-for-byte, SHA-256 `344c4287f855ce7ee01aca2bfd386589e9f8f316832c55799e58e96b08b036fe`. All 19 code cells have execution counts, 40 saved outputs and zero saved errors. Code-cell sources match commit `51866463c6064bf08b38b79a65875949bf351163`, tutorial blob `fa421bc43ff710f1c76fba3e4b6223c0138ba58d`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`8b86f63` (Created using Colab)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default path: six frozen backbones (ResNet-50, MobileNetV4-Conv-Small, ConvNeXt-Tiny, ViT-B/16, SwinV2-Tiny, EVA-02 Base 448) on the pinned iNaturalist CC0 six-species sample, split 108 train / 24 validation / 48 test; 5-NN and a linear probe (full-batch AdamW, lr 0.001, 1,000 epochs, lowest validation cross-entropy). BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, torchvision 0.29.0, timm 1.0.29, safetensors 0.8.0, NumPy 2.1.3 (preloaded by the host kernel, retained), CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 19; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Test 5-NN accuracy / probe accuracy / probe macro F1: ResNet-50 0.521 / 0.500 / 0.467, MobileNetV4 0.396 / 0.500 / 0.505, ConvNeXt-Tiny 0.771 / 0.729 / 0.727, ViT-B/16 0.812 / 0.792 / 0.789, SwinV2-Tiny 0.646 / 0.667 / 0.644, EVA-02 Base 448 0.833 / 0.896 / 0.896. Selected probe epochs: 35 (ResNet-50), 46 (MobileNetV4) and 34 (SwinV2); ConvNeXt, ViT and EVA-02 were selected at the 1,000-epoch cap and flagged `selected_at_epoch_cap`. No probe was selected at the first epoch. These values match a local CPU run of the same commit.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (BYOD and resolution-stress) or REL12 qualification gates (this notebook has no STANDARD/FULL tiers; SwinV2 and EVA-02 are part of the default run above), and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).


## Modern image workshop: Notebook Review Framework v1 findings — revision 0.2.0-candidate (2026-09-28)

A review under the Notebook Review Framework v1 examined commit `cf3dbad` (notebook blob `3083a52e`) and concluded **Needs revision**. It reported six major findings and five minor ones. The review and its probes are archived in [`reviews/2026-09-27-notebook-review/`](reviews/2026-09-27-notebook-review/). Revision `0.2.0-candidate` (notebook blob `fd59129eea28`) addresses every item.

`tests/test_modern_workshop_review_fixes.py` has 43 tests that execute the notebook's own cells: the 5-NN rule, the metric validation, probe fitting, SafeTensors export and artifact reconstruction, the BYOD loader and BYOD run, resolution stress, the validation-only activity and the diagnostic cells. A tiny fake backbone replaces the pretrained checkpoints. 42 of the tests fail on `cf3dbad`; the one that passes there is the committed-notebook-is-clean guard.

| Finding | Correction in 0.2.0 | Acceptance check |
|---|---|---|
| **M1** (major): 5-NN metrics used the argmax of the vote fractions, which ignores the cosine-sum tie-break | `classification_metrics` takes the model's own decision (`predicted_ids`); 5-NN passes its tie-aware decision, and the vote fractions stay unmodified as scores. The built-in and BYOD paths share one `knn_predict` | Unique majority, 2–2–1 ties (both directions), a five-way tie and equal summed similarity (resolved by lower class ID) are each scored by the decision. Real-model effect below |
| **M2** (major): ZIPs were flattened and reused a shared folder; CSV paths were not contained | The whole archive is validated first (unsafe paths, symbolic links, 1 GB limit, members that would land on the same file), then unpacked into a new notebook-owned folder that keeps its structure; a failed attempt's folder is removed. `labels.csv` paths must be relative, stay inside the dataset and avoid symbolic links. Cheap table checks run before any image is decoded. A user directory is never modified | Directory and ZIP give identical records with nested folders; a same-named unreferenced member cannot replace a referenced image; colliding members are refused; a complete-then-incomplete retry fails without reusing files; `..`, absolute, drive-letter and symlinked paths are refused; no image is decoded before the table checks pass |
| **M3** (major): the resolution-stress switch performed no experiment | Implemented: each test photo is reduced so its longer side is 96 or 160 px, the native transform enlarges it again, and the probe rebuilt from its saved artifact scores it. Nothing is refitted; each row records the probe SHA-256. When disabled, the notebook says it was skipped | Every model × {original, 96px, 160px} row exists with the saved probe's digest; no refit occurs; the backbone receives images whose longer side is 96 and 160 px; disabled mode produces no rows |
| **M4** (major): the "change one thing" activity displayed test results | A validation-only activity cell writes its plan first, fits probes on nested subsets, and reports validation loss and accuracy, the selected epoch, the training count and the change from the full set under `activity/`. `fit_probe` no longer requires test features. Section 15 is labelled a predeclared test curve | A sentinel that fails on any read of test features or the test split passes; the plan file exists; the activity is off by default |
| **M5** (major): diagnostics were computed but not shown | One training photo per species is shown before modelling. Section 11 explains every metric, with a worked accuracy-versus-log-loss contrast. Section 13 shows per-species recall, all six confusion matrices, and a deterministic gallery of real mistakes or disagreements (or says there are none). PCA plots are displayed, with explained variance and a caveat | Figures are shown, the gallery holds distinct real test photos in the hardest category first, "no mistakes" is stated rather than invented, and PCA is no longer closed without display |
| **M6** (major): reload verified tensors but not the serialized artifact | `load_probe_artifact` rebuilds the probe from `manifest.json` and `probe.safetensors` alone. It checks the format, base identity and the file's recorded size and SHA-256 before loading, and validates tensor shapes, normalisation and distinct classes. `reload_probe_verify` compares probabilities **and decoded species labels** with the class order the probe was trained on. BYOD uses the same functions | A reversed `class_order`, a false or changed probe digest, a missing manifest, a wrong dimension, a wrong base revision and duplicate classes are each rejected; an unchanged artifact reloads with difference 0.0 |
| Minor: probability validation | Scores of the wrong shape, non-finite or out of [0, 1], rows not summing to 1, and out-of-range class IDs are rejected before metrics | Fault injections, including every score set to 2.0 |
| Minor: BYOD disclosure | "Enforced requirements" replaces "recommended limits", with a point-of-use privacy warning ("not an on-premises system") and a description of what the outputs contain | Static checks |
| Minor: runtime guidance | Section 4 describes the kernel-Python install actually used. Troubleshooting says Runtime → Restart session (not delete runtime) and drops the "default tier" reference | Static checks |
| Minor: records | `predictions.csv` carries every class score; BYOD writes predictions, per-class metrics, provenance and a checked inventory; built-in and BYOD result scopes are labelled | BYOD run test checks the files, headers and literal labels |
| Minor: limits and latency | The limitations state the 48-image test size (about 2.1 points per photo), the observer overlap (now counted and exported) and possible pretraining overlap. Latency is named "backbone feature-forward" and its exclusions stated | Static checks |

### Real-model CPU pre-flight (2026-09-28)

The revised notebook was executed top to bottom on CPU with the real pinned corpus and all six pinned checkpoints. The runtime was Python 3.12, torch 2.14.0+cpu, timm 1.0.29, NumPy 2.5.3 and matplotlib 3.10.6. Resolution stress and the validation-only activity were switched on for this run. All 21 code cells completed without error; the script and results are `reviews/2026-09-27-notebook-review/cpu_preflight*`.

- **Probe results reproduced exactly:** probe accuracy, macro-F1 and the selected epochs equal the recorded 2026-09-26 Colab T4 run. The epochs are 35, 46, 1000, 1000, 34 and 1000, and the split digest is `842433b7…`.
- **The M1 correction changes the published 5-NN results.** With the new code on CPU, the old argmax rule reproduces the recorded values exactly. The tie-aware rule the specification requires gives:

  | Model | Recorded (argmax) | Corrected (tie rule) | Test rows with a tied vote | Decisions changed |
  |---|---|---|---|---|
  | ResNet-50 | 0.521 | **0.500** | 9 | 6 |
  | MobileNetV4-Conv-Small | 0.396 | **0.417** | 9 | 3 |
  | ConvNeXt-Tiny | 0.771 | **0.729** | 7 | 4 |
  | ViT-B/16 | 0.812 | **0.792** | 4 | 2 |
  | SwinV2-Tiny | 0.646 | **0.583** | 8 | 6 |
  | EVA-02 Base 448 | 0.833 | 0.833 | 0 | 0 |

  The historical figures above stay as recorded for their revision; revision 0.2.0 reports the corrected values.
- **Diagnostics:** 14 test photos were unanimous-correct, 19 majority-correct, 8 split and 7 shared hard cases. Recall ranges from 0.25 (several species for ResNet-50) to 1.00. The confusion matrices, the disagreement gallery and all PCA plots were rendered and saved.
- **Observer overlap:** 31 of 117 observers have photos in more than one split.
- **Resolution stress** (test accuracy at original → 160 px → 96 px):

  | Model | Original | 160 px | 96 px |
  |---|---|---|---|
  | ResNet-50 | 0.500 | 0.417 | 0.417 |
  | MobileNetV4-Conv-Small | 0.500 | 0.438 | 0.458 |
  | ConvNeXt-Tiny | 0.729 | 0.646 | 0.604 |
  | ViT-B/16 | 0.792 | 0.667 | 0.646 |
  | SwinV2-Tiny | 0.667 | 0.521 | 0.417 |
  | EVA-02 Base 448 | 0.896 | 0.688 | 0.667 |

  The `original` rows equal the main results, which confirms that the probes were reused unchanged.
- **Validation-only activity:** it produced the 6/12/18-per-class table without reading test data. With 6 photos per species, the validation-loss increase over the full set ranged from +0.035 (EVA-02) to +0.469 (ResNet-50).

- **BYOD with real backbones:** a ZIP built from the cached real photos was run through the BYOD cell with MobileNetV4-Conv-Small and ViT-B/16.
  - **Input:** the ZIP kept nested `birds/<species>/…` folders and used explicit 120/30/30 splits. The labels were `0`, `001`, `NA`, `house finch`, `American Goldfinch` and a 148-character Unicode label.
  - **Labels:** every label was preserved exactly in the probe manifests and in the `predictions.csv` score columns.
  - **Results:** 5-NN / probe accuracy was 0.567 / 0.467 for MobileNetV4 and 0.800 / 0.833 for ViT-B/16. Reload parity was 0.0 for both.
  - **Outputs:** the BYOD inventory (summary, predictions, class metrics, provenance and probes) was written and checked. The run is recorded in `cpu_byod_preflight*`.

Code cells changed, so the 2026-09-26 Colab record does not describe this revision. **Status: Candidate.** The following evidence is still required:

- a fresh Colab T4 default `Run all` of revision 0.2.0;
- the resolution-stress and validation-activity paths on the hosted runtime;
- a hosted BYOD run with a representative dataset.

The review's learner-observation recommendation remains open.

### Maintainer-supplied Colab execution of revision 0.2.0 — 2026-09-28

The maintainer supplied an executed Colab copy of revision `0.2.0-candidate`. It is preserved byte-for-byte as [evidence](execution-evidence/2026-09-28/DIMER_Modern_Image_Classification_Workshop.ipynb).

- **Reviewed source:** commit `cf5933b` (merged to `main` as `cfa94fa`), notebook blob `fd59129eea28`. All 50 cell ids, types and sources match the committed notebook exactly; no Colab `# @title` lines were added.
- **Executed-file SHA-256:** `93e9f52bf03deda8f5b8ba1941041cab8522f7edb3375fe069b752df1dd87275`.
- **Runtime:** Google Colab, Tesla T4 (`cuda:0`); Python 3.13.15, torch 2.14.0+cu130, torchvision 0.29.0, timm 1.0.29, safetensors 0.8.0, NumPy 2.1.3 (preloaded by the host kernel), Pillow 11.3.0.
- **Execution:** 21/21 code cells ran with execution counts 1–21 in order and no saved error outputs. The completion summary is present. The notebook's in-cell reload check raises on any difference, so the lack of errors means every probe artifact reloaded with matching probabilities and labels.
- **Configuration:** default `Run all` path. Resolution stress, the validation-only activity and BYOD were left off, and each cell says it was skipped.
- **Results:** split 108 / 24 / 48, split digest `842433b7…`, 31 of 117 observers in more than one split, majority floor 0.167.

  | Model | 5-NN acc | Probe acc | Macro-F1 | Log-loss | Selected epoch |
  |---|---|---|---|---|---|
  | ResNet-50 | 0.500 | 0.500 | 0.467 | 1.754 | 35 |
  | MobileNetV4-Conv-Small | 0.417 | 0.500 | 0.505 | 1.448 | 46 |
  | ConvNeXt-Tiny | 0.729 | 0.729 | 0.727 | 1.133 | 1000 |
  | ViT-B/16 | 0.792 | 0.792 | 0.789 | 0.897 | 1000 |
  | SwinV2-Tiny | 0.583 | 0.667 | 0.644 | 1.055 | 34 |
  | EVA-02 Base 448 | 0.833 | 0.896 | 0.896 | 0.367 | 1000 |

  These equal the real-model CPU pre-flight above to the printed precision, including the corrected tie-aware 5-NN values. The consensus counts also match: 14 unanimous-correct, 19 majority-correct, 8 split and 7 shared hard cases. The training gallery, confusion matrices, disagreement gallery and PCA plots rendered.
- **Evidence boundary:** saved outputs were inspected; the execution was not independently repeated. The separately exported files were not supplied.

This closes the "fresh Colab T4 default `Run all`" item above. The following remain open:

- the resolution-stress and validation-activity paths on a hosted runtime (covered by the CPU pre-flight only);
- a hosted BYOD run with a representative dataset (covered by the CPU pre-flight only);
- the learner-observation recommendation.

**Status: Candidate.**

### 2026-10-03 uv isolated environment — revision 0.3.0-candidate

Revision `0.3.0-candidate` moves the workshop off the in-kernel install: notebook blob `fd59129eea28` → `b8c151140103`. Two kernel setup cells now build a uv-managed CPython 3.12.12 environment from a carried hash lock (`tools/modern-image-workshop-requirements.lock`, 53 manylinux x86_64 wheels, installed with `--require-hashes --only-binary :all:`) and route every later cell to one persistent worker on that interpreter, so there is no install into the kernel and no restart step. The direct pins are unchanged; NumPy is now always the pinned 2.5.3 (the 2026-09-28 Colab run kept the kernel's preloaded 2.1.3). The notebook runs on **Linux x86_64 only**. Teaching cells, data, seeds, models and metrics are unchanged, and the two long JSON literals are split into pieces of at most 1,000 characters. `tools/build_modern_workshop_runtime.py --check` keeps the setup cell in step with the lock.

A local Linux x86_64 CPU run (WSL, GPU hidden) of this revision executed all 23 code cells through the real setup cells (uv wheel, managed Python, the full lock) and the worker with the real corpus and checkpoints, default configuration. Its 5-NN accuracy, probe accuracy, macro-F1, log-loss, selected epochs (35, 46, 1000, 1000, 34, 1000), split digest `842433b7…`, observer overlap (31 of 117) and consensus counts (14 / 19 / 8 / 7) equal the 2026-09-28 Colab T4 run of revision 0.2.0 to the printed precision. This is a builder pre-flight, not clean-runtime evidence. The 2026-09-28 Colab record above describes revision 0.2.0, not this one. **A hosted Colab T4 re-run of revision 0.3.0 is pending. Status: Candidate.**
