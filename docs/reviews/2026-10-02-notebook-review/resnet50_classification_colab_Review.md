# ResNet-50 Classification Tutorial Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch 2026-10-02)  
**Repository:** `kurtvalcorza/resnet50-classification-pipeline`  
**Notebook:** `tutorials/resnet50_classification_colab.ipynb`  
**Reviewed commit:** `86c899da3da6deb2eca14ec6af108ca5226d7a43` (`origin/main`)  
**Notebook Git blob:** `77dc7ebd3d7b3b479c21cdf6a6c7fd82c9048fd7`  
**Framework:** Notebook Review Framework v1 · **Requirements baseline:** NOTEBOOK_SPEC 2.2 (ml-worker `origin/main`); the notebook declares 2.0  
**Finding prefix:** `RN`

## Executive assessment

The inference half is solid. The pinned checkpoint is digest-verified before loading, the gradient sample is explained honestly as having no ground truth, the evaluation report says `not-measurable` and switches to `sample-sanity` when a class index is supplied, the argmax rule and uncalibrated softmax are stated correctly, the single-image BYOD limit (4096 px) matches the code, and the fine-tuned artifact reloads from files with `strict=True`. A local CPU run of this blob completed all 10 code cells in 24 s.

The fine-tuning half does not deliver what it claims. On the default data the fine-tuned model scores **0.50 accuracy against a 0.50 majority-class baseline** (it labels all 6 held-out images `truck`), and the report still says `verdict: success`. Nothing in the notebook asks the learner to notice or explain this. The prose calls the stage "head adaptation"; `fit` actually trains every parameter of the network. The dataset BYOD branch silently folds a user's `test/` folder into training, does not check class coverage, and accepts inputs that only fail after training. Install still happens in the live kernel with a restart-on-stale guard, and the notebook has almost none of the guided layer its `GUIDED` mode declares.

**Relation to the earlier review.** The 2026-09-27 review in `docs/reviews/2026-09-27-notebook-review/` covered only `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb` (it never mentions this notebook), and the 2026-10-03 uv move changed only that workshop. This notebook's blob has not changed since `b7e787d` (2026-09-14). None of the earlier findings apply here, so none is carried over or re-raised. This is the first Framework v1 review of this notebook.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Profile / mode | `E2E` / `GUIDED`; `standalone: true`; declares NOTEBOOK_SPEC 2.0 (current 2.2) |
| Stated learner | "basic Python and PIL image handling; what a softmax over class logits is" (cell 1) |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA used when present |
| Promised outcomes | Pinned install; carried module; digest-verified pinned snapshot; synthetic sample → input manifest (with a rejection finding); top-5 classification; evaluation report; CIFAR-10 subset download with digest check and seeded 80/20 split; bounded in-kernel fine-tune ("head adaptation"); artifact export; fresh reload; held-out evaluation against majority baseline; JSON/CSV outputs and provenance; two BYOD branches (single image, dataset ZIP) |
| Generator | `tools/build_notebook.py` + `tools/notebook_template.py`; module carried from `src/resnet50_classification_pipeline/pipeline.py` (sha256 `43b423cd…`, equal to the current file) |
| Existing execution evidence | `docs/release-verification.md`: Kaggle T4, 2026-09-14, commit `b7e787d` / blob `77dc7ebd3d7b`, **the same blob as reviewed**: "PASSED — 10/10 ok code cells executed cleanly". There's no archived executed notebook and no fine-tune or held-out numbers. The procedure in that file predates fine-tuning and checks only the inference stages. The Kaggle kernel `dimer-nb2-resnet50-classification` could not be opened from this review (permission denied). No Colab run recorded. No BYOD run recorded (REL12). Status `Candidate`. |

### Journeys and evidence basis

| Journey | Basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | Accurate inference narrative; guided scaffolding largely absent (RN-M4); the fine-tune result is never interpreted (RN-M2) |
| Clean default | Documented execution (Kaggle T4, same blob, outcome only) + direct execution (local CPU, labelled) | Hosted: passed per record, no numbers. Local CPU (install skipped via the cell's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; venv equal to `PINS` with CPU wheels; snapshot pre-copied, so the Hub download stage was **not** exercised; CIFAR subset downloaded for real): 10/10 cells, 24 s, 7 outputs. Top-1 on gradient `web site` 0.039; fine-tuned accuracy 0.50 = baseline 0.50 |
| Active learning | Direct execution (local CPU) | No exercise exists. Surrogates: rerunning cell 13 after the fine-tune gives the same top-5 (`pipe` is not mutated); rerunning 17→19 gives the same accuracy but a different loss (0.6964 vs 0.6929, augmentation is not seeded, RN-m3); `epochs=3` still 0.50; the `GROUND_TRUTH_INDEX` path switches the report to `sample-sanity` as documented |
| Reuse and recovery | Direct execution (local CPU, `google.colab` shim) | Single-image BYOD: 300×200 accepted, 4200 px rejected with the rule named. Dataset BYOD: unsplit and train/val ZIPs complete. A `train/`+`test/` ZIP is silently re-split. A val-only class is dropped. 1-image classes are accepted in explicit splits. A 4200 px training image fails only at evaluation, after training. A corrupt image gives a raw `UnidentifiedImageError`. An `interval/` folder empties val and crashes evaluation (RN-M5). Download failure and digest mismatch both end in `NameError: SAMPLE_HEIGHT` (RN-m1). Without Colab, both upload branches raise `ModuleNotFoundError` (RN-m2). Nothing was run in a hosted runtime |

**Limitations.** No GPU, no Colab/Kaggle run by this review, no learner observation. Local CPU execution is not hosted verification, and GPU kernels could move the fine-tune numbers slightly. The install cell and the Hub snapshot download were not exercised locally. The probe environment was an existing venv (`detr-detection-pipeline/.venv`, read-only) whose versions equal `PINS`. `torchaudio` was absent; the notebook never imports it.

## 2. Separate judgments

- **Technical correctness:** inference, validation, snapshot verification and artifact reload are correct (the reloaded model's scores equal the in-memory model's, max difference 0.0, though the notebook never checks this, RN-m4). The defects are the restart pattern (RN-M1), the broken fallback (RN-m1), and BYOD dataset handling (RN-M5).
- **Promise fulfilment:** the inference promises are met. "Head adaptation" is not what runs (RN-M3). The fine-tune stage runs but adapts nothing measurable on the default data (RN-M2). The "graceful" fallback does not work (RN-m1).
- **Scientific validity:** a 6-image validation set (one image = 16.7 points) is reported with a hard-coded `success` and no uncertainty. The BYOD `test/` folder leaks into training (RN-M5). No near-duplicates between train and val in the default subset (minimum 8×8 average-hash Hamming distance 14 of 64; 0 exact duplicates).
- **Learner experience:** clear inference explanations. Minimal guided layer, no activity, and the fine-tune result is never interpreted (RN-M2, RN-M4).
- **Spec conformance:** RUN10/ENV6 (MUST) at risk on Colab (RN-M1, inferred). FT5 (MUST) unmet: the trainable set is misstated. §21.1 class-coverage validation (MUST) unmet for BYOD. OUT8 (MUST) partial. VER5 (MUST) partial. REL5/REL10/REL12 (MUST) incomplete. GDL1–GDL15 (SHOULD) largely unmet. EXE2 (SHOULD) unmet. Declares 2.0.

## 3. Findings

### RN-M1 — Major: Install is into the live kernel; the documented Colab runtime is expected to stop for a restart

- **Cell/section:** cell 3, §1 "Install the pinned runtime".
- **Observed issue:** the cell runs `pip install` for `torch==2.14.0 … numpy==2.5.3 pillow==11.3.0` into the running kernel and raises `RuntimeError('Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.')` when a preloaded distribution changed. The markdown above the cell presents this as a feature.
- **Consequence:** on a host kernel that preloads NumPy at another version, Run all stops after the first code cell and the learner must restart and run again. The intro's one-pass Run all promise is not met.
- **Evidence:** source inspection (`tools/build_notebook.py:48-72`). Documented evidence from this repository: the 2026-09-26 Colab record of the sibling workshop shows "NumPy 2.1.3 (preloaded by the host kernel)", which differs from this notebook's `numpy==2.5.3` pin. The guard is therefore expected to fire on Colab, but that is **inferred, not observed for this notebook**. The only hosted record of this blob (Kaggle, 2026-09-14) reports a clean pass without saying whether a restart occurred. The same guard fired on Kaggle for a fleet sibling (raft-optical-flow, numpy 2.0.2 → 2.5.3).
- **Recommended correction:** adopt the uv isolated-environment pattern rather than a new install guard. A carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked requirements file with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env, so the kernel's preloaded NumPy/torch are never replaced. References: this repository's own `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb` (moved to uv in `26eb545`, lock in `tools/modern-image-workshop-requirements.lock`) and `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`. Change `tools/build_notebook.py` (install cell) and regenerate; drop the unused `torchaudio` pin while there.
- **Acceptance check:** a fresh Colab runtime executes the regenerated notebook top-to-bottom with Run all and no restart, recorded with blob id in `docs/release-verification.md`; no code cell can raise a restart instruction.
- **Spec:** RUN10, ENV6 (MUST).

### RN-M2 — Major: The default fine-tune learns nothing measurable, and the notebook reports it as `success`

- **Cell/section:** cell 17 (`pipe.fit(… epochs=1, batch_size=4, learning_rate=1e-4)`), cell 19 (evaluation), cell 22 (interpretation).
- **Observed issue:** on the default CIFAR-10 subset (16 per class: 26 train / 6 val, 32×32 images upscaled to 224) the fine-tuned model predicts `truck` for all 6 held-out images. Accuracy is 0.50, the majority baseline is 0.50, lift is +0.00, frog recall is 0/3, and mean confidence is 0.52. `finetuned_eval_report['verdict']` is hard-coded `'success'` (`tools/notebook_template.py:400`). The interpretation section never mentions the fine-tuned result. The section 8 prose explains neither the 6-image validation set nor the 32 px source images.
- **Consequence:** the central E2E demonstration, adapting the model to custom classes, shows no adaptation. The learner gets a `success` label and no prompt to compare against the baseline, so they are likely to conclude that fine-tuning worked, or be confused without guidance. With 6 validation images, each prediction moves accuracy by 16.7 points, and the notebook does not say so.
- **Evidence:** direct execution, local CPU (`results.json` → `default.finetuned_metrics`, `default.per_class`, `default.history`: train loss 0.696, val loss 0.691). Raising to `epochs=3` still gives 0.50 (`active.epochs3`; val loss rises 0.679 → 0.706). Hosted numbers are not recorded anywhere, so the T4 result is **not verified**; GPU kernels could shift it, but nothing here suggests a different outcome.
- **Recommended correction:** in `tools/notebook_template.py` (cells 17/19/22), choose a default configuration that demonstrably beats the baseline on the bundled data and verify it on the hosted runtime. Options: a frozen backbone with a trained head, as the prose claims (see RN-M3); more images per class; more steps or a higher head learning rate. Derive the verdict from the comparison instead of hard-coding it (e.g. `above-baseline` / `not-above-baseline`). Label the metrics as tutorial evidence, state the validation size and the per-image granularity, and add an interpretation checkpoint that asks the learner to compare accuracy with the baseline and with per-class results.
- **Acceptance check:** on a fresh hosted run of the regenerated notebook, held-out accuracy exceeds the majority baseline by more than one validation image. The report's verdict is computed, not literal. The interpretation section names the fine-tuned result, the baseline and the validation size.
- **Spec:** RUN7/FT2 (bounded adaptation must be real), FT7, EVAL6, GDL8, GDL14; framework dimensions 1, 3, 5.

### RN-M3 — Major: The notebook says "head adaptation"; `fit` trains the whole network

- **Cell/section:** cells 0, 16 and `tutorials/README.md`; module `fit` (`src/resnet50_classification_pipeline/pipeline.py:411`).
- **Observed issue:** the prose says "`pipe.fit(...)` implements 100% in-kernel head adaptation: it replaces the 1000-class head with a new linear classifier…", and the objectives say "classification head fine-tuning". `fit` builds `torch.optim.AdamW(model.parameters(), …)` with every parameter trainable and the model in `train()` mode, so BatchNorm running statistics are updated from batches of 4. All 318 shared tensors of the exported `model.safetensors` differ from the base snapshot, including `conv1.weight` and `bn1.running_mean`.
- **Consequence:** the learner is told the backbone is preserved when it is not. They cannot reason correctly about overfitting, compute cost or what the 94 MB artifact contains. The tiny-batch BatchNorm updates are a plausible contributor to RN-M2 (inferred).
- **Evidence:** source inspection, plus direct execution comparing exported tensors to the base snapshot (`results.json` → `default.fit_params`: `changed_vs_base` 318/318, `conv1_max_abs_delta` 6.7e-4).
- **Recommended correction:** decide which method the tutorial teaches. If head-only: freeze the backbone (`requires_grad_(False)` except `fc`) and keep BatchNorm in eval mode in `fit`. If full fine-tuning: rewrite the prose in `tools/notebook_template.py` (lines 66–76, 209–220). Either way, print the trainable and frozen parameter counts and record them in the exports.
- **Acceptance check:** the notebook prints trainable/total parameter counts, the prose names the same method, and comparing the exported tensors with the base snapshot confirms it (only `fc.*` changes for head-only).
- **Spec:** FT5, FT3 (MUST); ART2.

### RN-M4 — Major: Declared `GUIDED`, but the guided layer is essentially missing

- **Cell/section:** whole notebook (23 cells, 13 markdown).
- **Observed issue:** a static scan finds no "how to use this notebook", no roadmap, no glossary, no troubleshooting section, no "what to notice"/"expected result" notes, no prediction prompt, no interpretation checkpoint or sample answer, and no exercise. "Look for…" sentences appear in some stage intros. The learning objectives are mostly not observable ("install the pinned runtime", "read what the carried pipeline module guarantees"). The 510-line carried module cell (cell 5) is not labelled as infrastructure the learner can skip. The only "next experiment" for fine-tuning says to "supply your own multi-class dataset folder to `pipe.fit`", which contradicts the ZIP-based BYOD branch and gives no steps.
- **Consequence:** a self-paced learner is shown outputs but never asked to predict, interpret or diagnose them. They cannot tell which cells are learning material, and they have no recovery guidance when a hosted run fails (restart, download, memory, BYOD).
- **Evidence:** source inspection; `results.json` → `static.guided_layer_hits`.
- **Recommended correction:** in `tools/notebook_template.py`, add a "How to use this notebook" section with a roadmap (GDL2–3). Label cells 3/5/7 **Infrastructure** (GDL11). Rewrite the objectives as observable actions (GDL5). Add a prediction before classifying the gradient and before the fine-tune (GDL7). Add "What to notice" after each principal stage (GDL8) and 2–3 checkpoints with collapsible sample answers, e.g. why the gradient gets a confident-looking label, and whether the fine-tune beat the baseline (GDL9). Add one Predict → Change → Run → Observe → Explain activity with explicit rerun instructions, e.g. frozen vs full fine-tune or the number of images per class (GDL10, UX5). Add a glossary (logit, softmax, top-k, majority baseline, fine-tuning vs head training), a troubleshooting section (GDL13), and a conclusion template that covers the fine-tune result (GDL14).
- **Acceptance check:** each of GDL1–GDL14 can be pointed to a specific cell of the regenerated notebook, and `build_notebook.py --check` and the parity tests still pass.
- **Spec:** GDL1–GDL15, UX5, UX9 (SHOULD).

### RN-M5 — Major: Dataset BYOD silently re-splits user `test/` data, skips class-coverage checks, and fails late

- **Cell/section:** cell 17, BYOD branch (`tools/notebook_template.py:239-330`).
- **Observed issue:** split detection is a substring test (`'train/' in n`, `'val/' in n or 'valid/' in n`). Direct execution with representative ZIPs showed:
  - `train/` + `test/` (a common layout): `has_val` is false, so both folders are pooled and re-split 80/20. The user's test images go into training without any message.
  - `train/` + `val/` where val has a class absent from train (`bird`): those images are dropped silently. When train has a class absent from val (`dog`), the report shows `dog: total 0, accuracy 0.0` and a majority baseline of 1.0.
  - Explicit splits with 1 image per class are accepted. The ≥2-per-class rule applies only to the unsplit path.
  - A training image larger than 4096 px is accepted and trained on, then evaluation fails afterwards with the `MAX_IMAGE_SIDE` error.
  - A corrupt image raises a raw `UnidentifiedImageError` that does not name the file.
  - A folder named `interval/` matches `'val/'`, leaves 0 validation images, and evaluation crashes with `max() iterable argument is empty`.
  - No archive size, image count or per-image limits are stated or enforced. Prerequisites cover only the single-image branch, although cell 0 says dataset limits are stated there.
  - Path traversal (`..`) and non-ZIP uploads are refused with clear messages, and the unsplit 1-class and 1-image-class cases give rule-naming errors.
- **Consequence:** a user's held-out test data can leak into training, which inflates their result, and metrics can be computed over classes the model never saw. These are the leakage and coverage errors the notebook should guard against. Some failures arrive only after the expensive fine-tune.
- **Evidence:** direct execution, local CPU with a `google.colab` shim (`results.json` → `reuse.*`).
- **Recommended correction:** match split directories by path component (`parts[0] in {'train','val','valid','test'}` or a documented root). Preserve a supplied `test/` split, or refuse it with a message, rather than pooling it. Require every class to appear in both train and val, with a minimum count, before `fit`. Run `validate_inputs`-equivalent checks (size ceiling, decodability with the filename) on every image before training. State the dataset limits and enforce an expanded-size cap (§20).
- **Acceptance check:** each ZIP in `run_probes.py` → `byod(...)` is either refused before `fit` with a message naming the rule and the file/class, or (for valid layouts) completes. A `train/`+`test/` ZIP never moves a `test/` image into `train_images`.
- **Spec:** §21.1 class coverage (MUST), DAT12, DAT13, DAT19, VAL1, SPL2, SPL5; §20 expanded-size limit (SHOULD).

### RN-m1 — Minor: The advertised "graceful" synthetic fallback crashes, and a digest mismatch is routed into it

- **Cell/section:** cell 17, `except Exception` around the download (template line 257); cell 16 prose "gracefully falls back to deterministic synthetic stripes".
- **Observed issue:** both a failed download and a SHA-256 mismatch print a warning and enter the fallback, which uses undefined `SAMPLE_HEIGHT`/`SAMPLE_WIDTH` (only `SAMPLE_SIDE` exists) and raises `NameError`. Had the fallback worked, an integrity failure would have silently switched the tutorial to a different dataset.
- **Consequence:** when the Hugging Face dataset is unreachable, the learner gets a confusing `NameError` instead of an actionable message. The digest check does not fail closed on its own terms.
- **Evidence:** direct execution (`results.json` → `reuse.download_failure`, `reuse.digest_mismatch`); static (`static.SAMPLE_HEIGHT_defined_anywhere: false`).
- **Recommended correction:** fail closed on a digest mismatch with a message naming expected and actual digests. Either fix the fallback (`SAMPLE_SIDE`) and label its results as synthetic in every report, or remove it and its prose.
- **Acceptance check:** a forced digest mismatch raises an error naming both digests. A forced download failure either completes on the labelled synthetic set or stops with an actionable message, never a `NameError`.
- **Spec:** DAT2, UX10; MOD8 (by analogy for data).

### RN-m2 — Minor: Both upload branches are Colab-only

- **Cell/section:** cells 9 and 17.
- **Observed issue:** `USE_BYOD` and `USE_BYOD_DATASET` call `google.colab.files.upload()` with no location field. On Jupyter or Kaggle they fail with `ModuleNotFoundError: No module named 'google'`.
- **Evidence:** direct execution (`reuse.upload_without_colab`, `reuse.single_image_upload_without_colab`).
- **Recommended correction:** add `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH` form fields. When set, read from the path; otherwise use the Colab upload, and outside Colab say to set the path.
- **Acceptance check:** with the path fields set, both branches run outside Colab without importing `google.colab`.
- **Spec:** EXE1, EXE2.

### RN-m3 — Minor: Fine-tune configuration is not recorded, and the run is not reproducible

- **Cell/section:** cells 17, 21; `fit` defaults.
- **Observed issue:** the exports record only `history` and `classes`. `model-config.json` has architecture, classes, model id/revision and data config, but no optimizer, learning rate, epochs, batch size, weight decay, seed or trainable set. `fit` seeds torch with its own default `20260910`, not the notebook's `SEED = 42`, and timm's training transforms draw from Python's unseeded `random`. Rerunning cells 17→19 gives a different loss curve (0.6964 vs 0.6929).
- **Evidence:** direct execution (`default.model_config_keys`, `default.result_json_fine_tuning_keys`, `active.rerun_history`).
- **Recommended correction:** pass `seed=SEED` and seed `random`/`numpy` in `fit`. Write the full adaptation configuration and the trainable-parameter count into `model-config.json` and the result JSON. State the remaining variability.
- **Acceptance check:** two consecutive 17→19 runs give identical histories on CPU, and the exported config contains every FT6 field.
- **Spec:** OUT8 (MUST), FT6, ENV7, ENV8.

### RN-m4 — Minor: Reload proves loading, not equivalence

- **Cell/section:** cell 19.
- **Observed issue:** `fine_tuned_pipe` (the in-memory model) is never used after `fit`. The reloaded pipeline is evaluated on its own, so the notebook does not show that the artifact reproduces the trained model's outputs.
- **Evidence:** static (`static.fine_tuned_pipe_uses: 1`, the assignment only). The probe found the two equal (max score difference 0.0), so this is a missing demonstration, not a defect in the artifact.
- **Recommended correction:** compare in-memory and reloaded scores on the validation images with an explicit tolerance and print the result.
- **Acceptance check:** the notebook prints a max absolute score difference and a pass/fail against a stated tolerance.
- **Spec:** VER4 (SHOULD), VER5 (MUST).

### RN-m5 — Minor: Execution record does not cover the fine-tune stages, and status documents contradict each other

- **Cell/section:** `docs/release-verification.md`, `STATUS.md`, `tutorials/README.md`.
- **Observed issue:** the only hosted record of this blob (Kaggle T4, 2026-09-14) gives an outcome but no numbers, and no executed notebook is archived. The verification procedure (step 5) lists only inference stages, with no fine-tune, reload or held-out evaluation. "Current status" says "No clean-runtime execution of the notebook has been recorded yet; clean GPU execution evidence is now recorded below". `STATUS.md` says "awaiting clean-runtime execution". `tutorials/README.md` says Run-all is "verified". No Colab run (the documented user runtime) and no BYOD run is recorded.
- **Evidence:** source inspection; the Kaggle kernel could not be opened (permission denied).
- **Recommended correction:** extend the procedure to the fine-tune, reload and evaluation stages. Archive the executed notebook with its outputs. Record a Colab run and a BYOD run, and reconcile the three status statements.
- **Acceptance check:** `docs/release-verification.md` holds a record for the reviewed blob with fine-tuned accuracy, baseline and reload result, plus a REL12 BYOD record. The three documents state the same status.
- **Spec:** REL5, REL10, REL12 (MUST).

### RN-m6 — Minor: Declares NOTEBOOK_SPEC 2.0; current is 2.2

- **Cell/section:** cell 0, metadata, `tools/build_notebook.py:31`, `tools/validate_release_assets.py:98`.
- **Recommended correction:** migrate to 2.2 when regenerating and update the validator constant.
- **Acceptance check:** metadata, cell 0 and the validator all name 2.2.
- **Spec:** §30 conformance reporting, §32 change control.

### Suggestions

- **RN-S1:** remove the `torchaudio==2.11.0` pin. No cell imports it, and it adds install time and a version-coupling risk with `torch==2.14.0`.
- **RN-S2:** show a few training images per class and explain that CIFAR images are 32×32 upscaled to 224×224. This helps learners interpret RN-M2 and choose BYOD data.

## 4. Positive findings and non-findings

- No unconditional sample asserts. Rerunning classification after fine-tuning is safe: `fit` is a classmethod that builds a new model, and `pipe` stays pretrained (`active.cell13_rerun_after_fit_same_top5`).
- The single-image BYOD limit in the prerequisites (≤4096 px) matches `MAX_IMAGE_SIDE`, and oversize is rejected with the rule named.
- No train/val near-duplicates in the default subset.
- No CPU or stale numbers are quoted as facts. The upstream 80.38 %/94.60 % is labelled as quoted, not measured.
- The decision-rule and calibration statements are accurate. The zero-shot evaluation report is honest about not being measurable.

## 5. Readiness

**Needs revision.** Five Major findings are open (RN-M1 to RN-M5). FT5, §21.1 class coverage, OUT8, VER5 and REL5/REL10/REL12 MUSTs are unmet, and RUN10 is at risk on Colab. Suggested order: RN-M3 together with RN-M2 (decide the method, then make the demonstration work), RN-M1 (uv), RN-M5, RN-M4, then the minors. A hosted Colab run of the regenerated notebook and a BYOD run are the remaining gates.

## 6. Verified versus inferred

- **Verified by direct execution (local CPU):** default-path completion; the fine-tune result (0.50 vs 0.50); full-network updates; BYOD behaviours; the fallback `NameError`; non-determinism of the fine-tune; reload equivalence (0.0).
- **Documented only:** the hosted Kaggle pass of this blob (outcome only).
- **Inferred:** that the install guard fires on Colab (RN-M1), and that hosted fine-tune numbers resemble the CPU ones (RN-M2).
- **Not verified:** any hosted run by this review, the Hub download stage, and learner understanding.
- **Most likely to be wrong:** RN-M2's hosted outcome. A T4 run could land one image above baseline by chance. Even so, the hard-coded `success` and the missing interpretation would still stand.

Probe bundle: `resnet50_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
