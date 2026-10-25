# Release verification

`tutorials/resnet50_classification_colab.ipynb` (`E2E`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, `standalone: true` and `generated_from` (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/resnet50_classification_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; exactly two kernel cells — the isolated install (pinned `uv` wheel checked by size and SHA-256, managed
  CPython, `--require-hashes --only-binary :all:` from the carried hash lock) and the router — with every later cell
  routed to the isolated environment; `NOTEBOOK_SOURCE` recorded in exports;
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
- the review fixes (RN-M1..M5, RN-m1..m6): the fine-tuning configuration, trainable set and dataset provenance are
  printed and exported, the held-out verdict comes from `finetune_evaluation_report` (no literal verdict), the
  fallback and BYOD archives go through `synthetic_stripes_dataset` / `load_image_zip`, the reload is compared with
  the in-memory model, stale learner-facing claims are absent, and the guided layer (who it is for, how to use,
  roadmap, predictions, *What to notice*, worked answers, the Section 12 activity, troubleshooting, glossary,
  conclusion; Infrastructure titles on the five setup cells) is present;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchvision` wheels plus `timm`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab runtime, Linux x86_64 (T4 GPU or CPU; CUDA used automatically when present) | The runtime the tutorial is written for; a one-pass top-to-bottom `Run all` here is promotion evidence |
| Colab CLI / Kaggle kernel | Fresh Linux x86_64 VM (Tesla T4) | Clean-room executor of the same class; the notebook is executed verbatim (Kaggle: plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory). No repository checkout is needed — the notebook is standalone |
| Local harness (pre-flight only) | Workstation, sequential cell executor with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` and a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

The notebook supports **Linux x86_64 runtimes only**: its isolated environment is built from manylinux wheels, and
Section 1 stops with that message on Windows or macOS.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new Linux x86_64 runtime (Colab, or the executors above) with **no
   repository checkout** and a clean model cache;
3. choose **Run all once** without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`, `USE_BYOD_DATASET = False`, `TRAINABLE = 'head'`); a run that needs
   a restart or a second pass is not a `Run all` PASS and must be recorded as such;
4. verify that Section 1 builds the isolated environment (`isolated_python` 3.12.12, the locked package count) and
   that the runtime cell reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and the installed core package versions equal to the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - pinned `timm/resnet50.a1_in1k` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/resnet50-a1/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - synthetic 256×256 gradient sample generated in code with its RGB SHA-256 printed and the
     ceilings (`NUM_CLASSES` 1000, `MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 64) surfaced;
   - `validate_inputs` writes `outputs/resnet50_classification_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the oversized probe);
   - classification through `predict(image, top_k=5)` with `decision_rule == 'argmax'` and a
     rank-ordered top-5 list;
   - `evaluation_report` writes `outputs/resnet50_classification_evaluation_report.json` with verdict `not-measurable`
     on the synthetic sample (no ground truth), stated as such;
   - Section 8: the tutorial dataset is downloaded and digest-verified (`66f90a4f…`, not the synthetic fallback), with
     a pair-grouped split of 80 training and 20 held-out images per class (both copies of a photo on one side) and
     `held_out_sharing_a_file_name_with_train: 0`;
   - Section 9: `head-only fine-tuning` with 4,098 trainable and 23,508,032 frozen parameters printed before training
     (BatchNorm statistics frozen),
     and the one-epoch history;
   - Section 10: the reloaded artifact reports `source == 'fine-tuned-artifact'` and `equivalent: True`, and the
     held-out report prints `n = 40`, the count, the 95 % Wilson interval, the majority baseline, the verdict
     `sample-sanity` and `comparison_to_baseline`;
   - `outputs/resnet50_classification_result.json` (with `fine_tuning.config`, `fine_tuning.dataset`,
     `fine_tuning.evaluation` and `fine_tuning.artifact_sha256`), `outputs/resnet50_classification_top_k.csv`,
     `outputs/resnet50_classification_validation_predictions.csv`,
     `outputs/resnet50_classification_finetuned_evaluation_report.json` and
     `outputs/resnet50_classification_finetuned/{model.safetensors,model-config.json}` (with `fine_tuning` in the
     config) are written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. exercise the BYOD gates (REL12): one compatible and one incompatible dataset archive through Section 8 (the
   incompatible one must be refused before training with the file or class named), and optionally one BYOD image;
8. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device),
   model identifier and immutable revision, whether the model cache was clean, `restarted: false`, outcome, the
   held-out count with its interval, produced outputs, and any warning or applicable `SHOULD` deviation in the table
   below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/resnet50_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/resnet50_classification_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-25 | `e5f1e56` / `1c89a8be` (PR #15 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `e5f1e56afea4ea1dc2b70bb18415572ea0d5405b`, git blob SHA-1 `1c89a8be310ce7101f2eb0a15a61bb3a01b4ad67` checked, input unchanged; captured executed-notebook SHA-256 `7f0f8814d678a52ac8c340c0eede045ce927bd593d701359ac2669c41a14d016` equals `run_summary.json` `executed_sha256` | 83.7 s | **PASSED in one pass, no restart, 0 errors** — 14/14 code cells (`exec.log`); synthetic image top-1 `web site, website, internet site, site` 0.0392, then `theater curtain` 0.0382, `matchstick` 0.0321, `dugong` 0.0221, `cleaver` 0.0209 (verdict `not-measurable`); head-only fine-tuning of 4,098 parameters (23,508,032 frozen), one epoch: training loss 0.6823, validation loss 0.6285, validation accuracy 0.85; held-out 34/40 = 85.0 %, 95 % Wilson 70.9 % to 92.9 %, `above-baseline` (frog 14/20, truck 20/20; six frog images called `truck`, the first at score 0.514); fresh-boundary reload agreed on 40 of 40 labels, `max_score_difference` 0.0. Not exercised: BYOD (REL12), the split-seed and full fine-tune experiments and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25-e5f1e56/` |
| 2026-09-14 | `b7e787d` / `77dc7ebd3d7b` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-resnet50-classification` v1) | Default sample path of the previous notebook version (in-kernel install, 16 images per class) | 182.3 s | **Restart status not recorded; not promotion evidence.** Recorded at the time as "10/10 ok code cells executed cleanly, 8 files, 103 MB staged"; no executed notebook and no fine-tuning numbers were archived, and the kernel could not be re-opened for review (permission denied). That blob pip-installed into the live kernel behind a restart-on-stale-import guard, so whether a restart occurred is unknown. Superseded by the review fixes (isolated environment, no restart); the current blob has no hosted run yet. |
| 2026-10-04 | `26eb545` / `b8c151140103` | Google Colab (browser, maintainer-run), Tesla T4 | `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb` revision 0.3.0-candidate, default `Run all` (toggles at defaults) | not recorded (setup 65 s) | **PASSED** — 23/23 code cells, counts 1–23, 0 errors; metrics equal the 2026-09-28 run ([record](#maintainer-supplied-colab-execution-of-revision-030--2026-10-04)) |
| 2026-10-09 | `72a6cda` / `75e7e1ee065d` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-resnet50-72a6cda-8a5f`) | Default path only (`TRAINABLE = 'head'`; inference, head-only fine-tuning, reload, held-out evaluation) | 86.5 s | **One pass, no restart, 0 errors**, 14/14 code cells. Held-out **34/40 = 85.0 %, 95 % Wilson 70.9 %–92.9 %**, above the 50 % majority baseline (verdict `sample-sanity`, `above-baseline`). See the record below. |
| 2026-10-15 | `4112eb3` / `037dcdebfe8b` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-resnet50-4112eb3-3dce`) | `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb` revision 0.4.0-candidate, default path (toggles at defaults) | 462.7 s (session) | **PASSED** — 23/23 code cells in order, 0 errors; every computed metric equals the 2026-10-04 run; 0 observations cross splits ([record](#colab-cli-execution-of-revision-040--2026-10-15)) |

### 2026-10-09 — Colab CLI one-pass run of `72a6cda` (fresh Colab Tesla T4)

- **Commit / notebook blob:** `72a6cda3e3c4fec7d18404ef6c33ca8ada9f7e6b` / `75e7e1ee065d7dfd7a0cf2724d47b87889818dd2`
  (blob checked against the fetched bytes before the VM was allocated; executed code-cell sources equal the commit's).
- **Executor:** Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab VM, Tesla T4, no repository
  checkout, clean model cache. This is **not** a browser `Run all`: the CLI sets no execution counts, so cell order is
  evidenced by `exec.log` (`Executing cell 1/14` … `14/14`, in order); forms were not rendered.
- **Path:** default settings only (`USE_BYOD = False`, `GROUND_TRUTH_INDEX = -1`, `USE_BYOD_DATASET = False`,
  `TRAINABLE = 'head'`). Wall time 86.5 s (suite wall clock, including the isolated install and the model download).
- **Outcome:** **one pass, no restart, 0 errors**; 14/14 code cells; cell 4 (the carried module definition) prints
  nothing by design. One Pillow `DeprecationWarning` (`mode` parameter) in the sample cell.
- **Runtime:** isolated Python 3.12.12 (kernel 3.13.15), 45 locked packages, setup 50 s, torch 2.14.0+cu130,
  timm 1.0.29, `cuda: True`, device `cuda:0`; `NOTEBOOK_SOURCE.repository_revision` `3322040e` =
  `metadata.dimer.generated_from` (the generation-time HEAD).
- **Model:** `timm/resnet50.a1_in1k` @ `767268603ca0cb0bfe326fa87277f19c419566ef` (apache-2.0, 3 files,
  102,509,031 bytes), all three fetched, verified and loaded with `source == 'local-snapshot'`.
- **Inference:** synthetic 256×256 gradient (RGB SHA-256 `e38db00b…`), input manifest `accepted` with the oversized
  probe `rejected`; `decision_rule == 'argmax'`, top-1 `web site, website, internet site, site` (916) at 0.0392;
  evaluation report `not-measurable` (no ground truth).
- **Section 8:** tutorial dataset `Cleanlab/cifar-10-subset @ bb5a7aab`, SHA-256 `66f90a4f…`; pair-grouped split
  80 train / 20 held out per class (160 / 40), `held_out_sharing_a_file_name_with_train: 0`.
- **Section 9:** head-only fine-tuning, 4,098 trainable / 23,508,032 frozen parameters, 1 epoch, batch 4,
  lr 1e-4; train loss 0.6823, val loss 0.6285, val accuracy 0.85.
- **Section 10:** reload `source == 'fine-tuned-artifact'`, 40/40 label agreement, max score difference 0.0,
  `equivalent: True`. Held-out **34/40 = 85.0 %**, 95 % Wilson interval **70.9 % to 92.9 %**, majority baseline
  50.0 % (`frog`), verdict `sample-sanity`, `comparison_to_baseline: above-baseline`; frog 14/20, truck 20/20;
  misclassified: both copies of `frog/image_42.png`, `frog/image_9.png` and `frog/image_65.png` (scores 0.50–0.52).
  These equal the worked answers' local CPU figures (loss 0.68 / 0.63, 34/40, interval 71–93 %, gradient top-1
  `web site` 0.0392).
- **Exports:** the result JSON, top-k CSV, validation-predictions CSV, input manifest, both evaluation reports and
  `resnet50_classification_finetuned/{model.safetensors,model-config.json}` were written.
- **Evidence files** (`docs/execution-evidence/2026-10-09-72a6cda/`, byte-for-byte from the run directory):
  - `resnet50_classification_colab_72a6cda_colab-cli-t4_output.ipynb` — SHA-256 `5685e628f60385a389465448bb72d08b2fd4a74c5287f44f7869905ca9f8c05c`
  - `exec.log` — SHA-256 `c3dcfc2e41d12e2644f6fb2819b6e06c8cc75964f34c90c41e1305a078e5b543`
  - `run_summary.json` — SHA-256 `7263c9b9969b9aa8e9c2b1bf79c2fb942638110743e04e053f34319fe549dddf`
- **Not exercised:** the BYOD image and BYOD dataset gates (step 7), the Section 12 activity (`TRAINABLE = 'all'`)
  and any other optional journey; no browser interaction.

## Current status

**Pending (prose kept true across runs, 2026-10-25):** the current blob is `tutorials/resnet50_classification_colab.ipynb` `8cb6ad08543807f7a8b8921df9e8e0e35e591e6c` (regenerated at `b117cd6`; markdown only). The Section 9 worked answer credited the `SPLIT_SEED` 0/1/2 scores (34/40, 37/40, 32/40) to the hosted Colab T4 run of `75e7e1ee`; the hosted runs used the default split only, so it now credits the 2026-10-09 local Windows CPU check, as Section 8 and the summary already did. The blob is **Candidate**; its hosted run is **pending**. The 2026-10-25 run of `1c89a8be` described next is history for the previous blob (REL14).

**Previous blob (2026-10-25): Candidate.** The previous notebook `tutorials/resnet50_classification_colab.ipynb` is blob `1c89a8be310ce7101f2eb0a15a61bb3a01b4ad67` (commit `e5f1e56`), regenerated from generator commit `8acff6e` for the tier C review findings of row 75. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 14/14 code cells, 83.7 s wall; held-out 34/40, 95 % Wilson 70.9–92.9 %, `above-baseline`; reload exact; evidence in `docs/execution-evidence/2026-10-25-e5f1e56/`) — the same figures as the run of blob `75e7e1ee`. The paths this blob's runs cover: the default path of this tutorial only (the workshop notebook is tracked separately). The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). Every run recorded for earlier blobs is history (REL14). Changes: T75-1 (REL14/REL13) replaces "no run of this version is recorded yet" with the Colab T4 run of blob `75e7e1ee` (`72a6cda`, 2026-10-09: 86.5 s, 34/40) and names it in the worked answers, keeping the local CPU check as the labelled source for the optional full fine-tune and split-seed figures; T75-2 (DAT9) states that frog and truck are ImageNet-1k classes of the frozen backbone (timm synsets) and that image-level overlap of the CIFAR-10 images was not checked; T75-3 states that the 40 held-out images are also the per-epoch validation set; T75-4 declares Notebook Specification 2.5 with `notebook_kind: TUTORIAL`, a use notice, IEEE references and the documented `DIMER_ISOLATED_ENV`. Local checks (generator `--check`, `tools/validate_release_assets.py`, ruff, pytest against the `main` baseline) are source checks, not clean-runtime evidence.

Evidence files for the 2026-10-25 default run of blob `1c89a8be`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-e5f1e56/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `resnet50_classification_colab_e5f1e56_colab-cli-t4_default.ipynb` | `7f0f8814d678a52ac8c340c0eede045ce927bd593d701359ac2669c41a14d016` | `7f0f8814d678a52ac8c340c0eede045ce927bd593d701359ac2669c41a14d016` (unchanged, no local path) |
| `exec.log` | `ceb05190395ea4edd90afb9e442ee1512b91b27ea75fa497c8932ae5452650f5` | `4448971fb0e7cd5f2256b3098eba2e24e096f350c46203d0e7b6a45397798eba` (2 local-path replacements) |
| `run_summary.json` | `03a58740ea17c6eb8c8196c50473db98a2fcd19a6ff92eac37937fd6c8673773` | `03a58740ea17c6eb8c8196c50473db98a2fcd19a6ff92eac37937fd6c8673773` (unchanged, no local path) |

**Prose check (2026-10-25).** Every number the notebook prose quotes from a recorded run was compared with what this run printed; none differs (timings and CPU-only figures that name another blob or runtime are labelled history and were not compared).

**Previous blob — Candidate.** The review fixes (RN-M1..M5, RN-m1..m6, review PR #12; mechanism ported from the sibling
`convnext-classification-pipeline` at `06482ea`) regenerated the notebook: no in-kernel install (the fleet's uv
isolated environment), head-only fine-tuning by default with the backbone and its BatchNorm statistics frozen and
the configuration exported, a held-out verdict computed from the counts with its interval on a 100-per-class
pair-grouped split, a validated dataset intake that refuses a `test/` folder and train/val class mismatches before
training, a working fallback, a reload equivalence check and the guided layer. Static validation, the generator
`--check`, ruff and the offline unit suite pass on this source. A local Windows CPU check of every code cell with the
real pinned weights (2026-10-09, not clean-runtime evidence) gave held-out 34/40 (Wilson 70.9 %–92.9 %,
`above-baseline`) for head-only and 21/40 (`indistinguishable-from-baseline`) for the Section 12 full fine-tuning.
A one-pass hosted run of the current notebook blob `75e7e1ee` is recorded above (2026-10-09, Colab CLI sequential
execution on a fresh Tesla T4, default path, 14/14 cells, no restart, held-out 34/40, Wilson 70.9 %–92.9 %).
Remaining gates: the BYOD release gate (step 7) on a hosted runtime and a reviewer's confirmation of the recorded
run against the blob under review. The registry status remains **Candidate** until an integrator promotes it;
promotion is not performed by the builder.

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

### Maintainer-supplied Colab execution of revision 0.3.0 — 2026-10-04

The maintainer ran revision `0.3.0-candidate` in a browser Colab session and supplied the executed copy. It is preserved byte-for-byte as [evidence](execution-evidence/2026-10-04/DIMER_Modern_Image_Classification_Workshop_26eb545_colab-browser-t4.ipynb).

- **Executed-file SHA-256:** `feadbdc5ef92040a9e4ec2fb16c87a20648fb62fcb71b68a5f5a3a1dab14d26a` (the committed copy has the same digest).
- **Source:** PR head `26eb545`, notebook blob `b8c151140103`. All 54 cell ids, types and sources match the committed notebook exactly. No `# @param` value was changed and Colab added no `# @title` lines.
- **Executor:** Google Colab (browser, maintainer-run), Tesla T4 (`cuda:0`, 15,637,086,208 bytes).
- **Runtime:** kernel Python 3.13.15. The uv setup cell built `/content/dimer_isolated_env` (CPython 3.12.12, 53 locked packages) in 65 s, with no restart. Every later cell ran in the worker on that interpreter: torch 2.14.0+cu130, torchvision 0.29.0, timm 1.0.29, safetensors 0.8.0, NumPy 2.5.3, Pillow 11.3.0, Matplotlib 3.10.6.
- **Execution:** 23/23 code cells ran with execution counts 1–23 in order, with no saved error outputs. The completion summary is present and 9 figures rendered. Total wall time is not recorded: the downloaded file has no per-cell timing.
- **Configuration:** default `Run all` path. Resolution stress, the validation-only activity and BYOD stayed off, and each cell says it was skipped.
- **Results:** split 108 / 24 / 48, split digest `842433b7…`, 31 of 117 observers in more than one split, majority floor 0.167. Consensus counts are 14 unanimous-correct, 19 majority-correct, 8 split and 7 shared hard cases.

  | Model | 5-NN acc | Probe acc | Macro-F1 | Log-loss | Selected epoch |
  |---|---|---|---|---|---|
  | ResNet-50 | 0.500 | 0.500 | 0.467 | 1.754 | 35 |
  | MobileNetV4-Conv-Small | 0.417 | 0.500 | 0.505 | 1.448 | 46 |
  | ConvNeXt-Tiny | 0.729 | 0.729 | 0.727 | 1.133 | 1000 |
  | ViT-B/16 | 0.792 | 0.792 | 0.789 | 0.897 | 1000 |
  | SwinV2-Tiny | 0.583 | 0.667 | 0.644 | 1.055 | 34 |
  | EVA-02 Base 448 | 0.833 | 0.896 | 0.896 | 0.367 | 1000 |

- **Comparison with the 2026-09-28 Colab T4 run of revision 0.2.0:** a line diff of the normalised text outputs leaves only expected differences. These are the setup output (uv environment against the old in-kernel install, with NumPy 2.5.3 against the preloaded 2.1.3), the revision string, Hugging Face download progress bars and the HF_TOKEN warning, which are absent now, and backbone latencies. Every accuracy, macro-F1, top-3, log-loss and per-class recall, every selected epoch (35 / 46 / 1000 / 1000 / 34 / 1000), and the representation-geometry, data-efficiency and consensus outputs are identical to the printed precision.
- **Evidence boundary:** this covers the default path only. Saved outputs were inspected and the execution was not independently repeated. The separately exported files were not supplied.

This closes the pending hosted Colab T4 re-run of revision 0.3.0. The following remain open:

- the resolution-stress and validation-activity paths on a hosted runtime;
- a hosted BYOD run with a representative dataset;
- the learner-observation recommendation.

**Status: Candidate.**

### 2026-10-14 review fixes (MIC2) — revision 0.4.0-candidate

Revision `0.4.0-candidate` fixes the 2026-10-12 Notebook Review Framework v1 findings on revision 0.3.0 (review of
`da35b54`, blob `b8c151140103`): notebook blob `b8c151140103` → `9fde027676ac`. The workshop has no generator for these
cells, so the edits were made by an anchor-asserting script; the setup cell still matches the lock
(`tools/build_modern_workshop_runtime.py --check`).

- **MIC2-M1 (owner decision A: disclose and reword).** Section 2 now has a per-checkpoint table of supervised label
  sets and of which target species are classes in them. All six species are classes of ImageNet-12k, 21k and 22k
  (WordNet IDs checked against `timm/data/_info` at pytorch-image-models `3dda92a3`). ImageNet-1k has house finch
  exactly and a broader "junco, snowbird" class (n01534433), and none of the other four. ConvNeXt-Tiny (12k), ViT-B/16
  (21k) and EVA-02 (22k) were supervised on label sets containing all six. The opening boundary, Sections 1, 5, 12
  and 21 and the sample interpretation now say that a gap between the two groups is consistent with label-set
  exposure confounding the comparison, not proof of it, and no longer credit the ranking to architecture, size or
  resolution. `provenance.json` records each model's label sets.
- **MIC2-m1.** Sections 8 and 21 state that one fixed probe recipe is used for all six backbones and is not tuned per
  backbone, and what a pick at the 1,000-epoch cap means. Section 11 prints the selected epoch and a cap flag.
- **MIC2-m2.** Sections 11 and 22 print the number of correct test photos and a 95% Wilson interval per probe
  accuracy; the count is re-derived from the saved predictions and must equal the recorded accuracy. The Section 12
  exercise asks which checkpoints the 48 photos can tell apart, and Section 15 says the curve is noisy.
- **MIC2-m3 (DAT20).** A refused BYOD input (including an empty `BYOD_PATH`) prints one message naming the input and
  the rule and how to recover, with no traceback, and removes earlier BYOD results under `byod/`. Undecodable images
  and failed integrity checks still stop the cell.
- **MIC2-m4.** The notebook declares NOTEBOOK_SPEC 2.5, `notebook_kind` `WORKSHOP` and `notebook_profile` `E2E`
  (the old `profile` key is kept), and `tools/validate_release_assets.py` checks the declaration.

Every change is a disclosure, a display or the BYOD refusal path. No default-path computation changed, so the metrics
of the 2026-10-04 run are expected to be unchanged; that is inferred from the diff, not re-executed.

**Verification (not clean-runtime evidence).** Lint, the release-asset validator, both generator checks and the
unit tests pass on Windows CPU with CI's dependency set; new tests in `tests/test_modern_workshop_mic2_fixes.py`
execute the notebook's own cells with stand-ins. The review's five acceptance checks fail on the old blob and pass on
the new one. A hosted run of `0.4.0-candidate` has not been made; the BYOD refusal must also be seen in a hosted
browser run (REL12).

**Status: Candidate.**

#### Peer-review follow-up (same revision, blob `9fde027676ac` → `037dcdebfe8b`)

An independent peer review of the MIC2 fixes raised further points, which were addressed on the same branch:

- **Interval-overlap rule removed.** Section 11 labels the per-model intervals as nominal binomial (Wilson)
  intervals that assume independent photos. Because all six probes score the same photos, it adds a paired display:
  for the probe with the most correct photos against each other probe, the photos only one of them got right and an
  exact McNemar p-value, labelled illustrative and with a multiple-comparison note. Display only.
- **Validation activity.** The reported minimum validation loss is called a validation-selection diagnostic: the
  same validation photos chose the epoch.
- **Observation audit.** In the pinned sample, all 180 photos come from 180 distinct iNaturalist observations and
  180 distinct photo IDs, so no observation or photo crosses splits. A 64-bit difference-hash comparison of the 180
  pinned photos (fetched and verified by size and SHA-256) found no pair within Hamming distance 10. The split cell now
  stops if an observation spans two splits, and `provenance.json` records the count (0). The near-duplicate check was
  run offline for the review and is not run by the notebook.
- **Wording.** The notebook now says "same linear-probe recipe, with input width set by each backbone" and describes
  results as transfer under this protocol. A cap pick is described as "the best observed validation loss occurred at
  the final epoch; extending training might change the selection". The learning curve is described as one
  nested-subset illustration, without attributing dips to noise. Resolution stress is described as synthetic
  source-detail degradation, and its rows record the smallest and largest resulting longer side.
- **Provenance.** `provenance.json` records each model's resolved `timm` preprocessing config and feature interface.
- **BYOD refusal.** Nothing is deleted any more. A refusal is recorded in `byod_attempts/<time>-refused/status.json`.
  Earlier results are moved out of `byod/` into `byod_attempts/`, so `byod/` holds no results that could be read as
  current.
- **Validator.** `validate_workshop_declaration()` now checks spec, kind, profile and mode structurally. It reads them
  from the metadata, from the opening cell's header fields and from the provenance literal's syntax tree.
- **Glossary.** New entries cover the three splits, epoch selection, standardisation and softmax scores.

The data-efficiency subset order is unchanged, because changing it would change the curve's numbers. A seeded random
order within each class is a follow-up. No default-path computed number changed. This is inferred from the diff and
was not re-executed.

**Status: Candidate.**

### Colab CLI execution of revision 0.4.0 — 2026-10-15

The PR head was run on a fresh Colab VM with the Colab CLI. The executed copy is preserved byte-for-byte as
[evidence](execution-evidence/2026-10-15/DIMER_Modern_Image_Classification_Workshop_4112eb3_colab-cli-t4.ipynb).

- **Executed-file SHA-256:** `d9a0cda8fc13098a1cdf9f6a12fe4a91a689af6baf227feb181da4509977ac58` (the committed copy
  has the same digest).
- **Source:** PR head `4112eb303c40af7b428ada536efd3e0da60e82dd`, notebook blob
  `037dcdebfe8b23b1d9005e0243682ae3c22235a7` (checked against the fetched bytes before the VM was allocated). All 54
  cell ids, types and sources match the committed notebook; no `# @param` value was changed.
- **Executor:** Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab VM, Tesla T4 (`cuda:0`,
  15,637,086,208 bytes), session `suite-resnet50-4112eb3-3dce`, released after the run. This is **not** a browser
  `Run all`: the CLI sets no execution counts, so order is taken from the executor log (cells 1–23 in sequence).
- **Runtime:** kernel Python 3.13.15. The uv setup cell built `/content/dimer_isolated_env` (CPython 3.12.12, 53
  locked packages) in 47 s, with no restart; torch 2.14.0+cu130, timm 1.0.29, NumPy 2.5.3. Session wall time 462.7 s.
- **Execution:** 23/23 code cells ran in order with no error outputs (one definition-only cell has no output); 9
  figures rendered and the completion summary is present.
- **Configuration:** default path. Resolution stress, the validation-only activity and BYOD stayed off, and each
  cell says it was skipped.
- **Observation audit:** the split cell printed `{'observations': 180, 'observations_crossing_splits': 0}`.
- **Results:** split 108 / 24 / 48, split digest `842433b7…`, 31 of 117 observers in more than one split, majority
  floor 0.167, consensus 14 unanimous-correct / 19 majority-correct / 8 split / 7 shared hard cases. Probe accuracy
  ResNet-50 0.500 (24/48), MobileNetV4-Conv-Small 0.500 (24/48), ConvNeXt-Tiny 0.729 (35/48), ViT-B/16 0.792 (38/48),
  SwinV2-Tiny 0.667 (32/48), EVA-02 Base 448 0.896 (43/48); selected epochs 35 / 46 / 1000 / 1000 / 34 / 1000. The
  paired table (EVA-02 against each other probe, only-EVA-right / only-other-right) prints 19/0, 21/2, 9/1, 6/1 and
  14/3, consistent with the correct counts.
- **Comparison with the 2026-10-04 browser run of revision 0.3.0 (blob `b8c151140103`):** a line diff of the text
  outputs leaves only expected differences: the uv setup time (65 s → 47 s) and worker pid, the revision string
  (0.3.0 → 0.4.0), backbone latencies (run-to-run timing), and the outputs this revision adds (the observation audit,
  the Wilson-interval and epoch-cap table, the paired McNemar table, the pretraining-label fields in the resource
  rows and the final summary's interval lines). Every 5-NN accuracy, probe accuracy, macro-F1 and selected epoch, and
  the consensus counts, are identical to the printed precision, which confirms the earlier inference that no
  default-path computed number changed.
- **Evidence boundary:** CLI sequential execution of the default path, not a browser `Run all`. Forms and uploads
  were not exercised; the BYOD accept and refuse paths, resolution stress and the validation activity were not run.

This closes the pending hosted re-run of revision 0.4.0 on the default path. The following remain open:

- a hosted browser BYOD run (accept and refuse, REL12);
- the resolution-stress and validation-activity paths on a hosted runtime;
- the learner-observation recommendation.

**Status: Candidate.**
