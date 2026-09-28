# DIMER Modern Image Classification Notebook — Review

**Verdict: Needs revision**  
**Review date:** 27 September 2026  
**Repository:** `kurtvalcorza/resnet50-classification-pipeline`  
**Notebook:** `tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb`  
**Reviewed commit:** `cf3dbadaade69dd320b6018832646601a2cd4df9`  
**Notebook Git blob:** `3083a52e48b7ae97523fb6073b3ba371e816c5fe`

## Executive assessment

The notebook implements a meaningful experiment: six pretrained vision backbones, the same bird-classification task, train-fitted feature normalization, a cosine 5-nearest-neighbor baseline, a common zero-initialized linear-probe policy, and held-out evaluation. Its distinction between comparing pretrained systems and isolating architecture is appropriate.

It is not yet a complete guided learning experience. An advertised resolution-stress control performs no experiment; a validation-oriented activity actually displays test results; and important diagnostic views are computed or saved without being presented to the learner. The 5-NN score also discards the implementation's intended tie-break. BYOD archive handling and artifact verification need substantive corrections.

Six Major findings below apply to different paths. They are not six proven failures of the default Colab run. In particular, this review does **not** establish that the recorded linear-probe accuracies are wrong, that test data trained the backbone, or that the hosted default run failed.

## 1. Review contract and evidence

The review applies the agreed **Notebook Review Framework — v1**: executable correctness, promise fulfillment, experimental validity, learner progression, interpretation, meaningful activity, interaction/recovery, and completion/transfer. A successful default run does not by itself verify the activity, BYOD, or artifact-consumption journeys.

| Item | Scope |
|---|---|
| Declared profile and mode | `E2E` / `WORKSHOP`; notebook specification `2.1` |
| Stated learner | A learner who can run Python notebook cells and is new to modern vision backbones, representations, or transfer learning |
| Default task | Six-species classification; pinned 180-photo corpus; 108/24/48 training/validation/test split |
| Default models | ResNet-50, MobileNetV4-Conv-Small, ConvNeXt-Tiny, ViT-B/16, SwinV2-Tiny, EVA-02 Base 448 |
| Additional default activities | Nested 6/12/18-image-per-class probe experiment; train-fitted PCA exports |
| Optional paths | Explicit-split BYOD and resolution stress |
| Adaptation | Frozen backbone; train-only standardization; CPU, full-batch AdamW linear probe; validation cross-entropy selection |
| Specification baseline | Declared 2.1; relevant requirements also checked against inspected fleet 2.2, dated 26 September 2026, Git blob `7428d5becb8d37133a3ad93a450d08ebf616f411` |

Sources: [opening and controls][N-open], [notebook methodology][N-method], [repository design specification][S], [fleet specification][F].

### Evidence actually obtained

**Source inspection:** Notebook code and markdown retrieved at the immutable commit, its design specification, and its release-verification record. This is a whole-learning-workflow review, not a PR-diff-only review.

**Repository execution evidence:** The supplemental notebook's release record documents a maintainer-supplied 26 September 2026 Colab execution: 19 executed code cells, 40 saved outputs, no saved errors, and terminal completion. The recorded code matches source commit `51866463c6064bf08b38b79a65875949bf351163`, notebook blob `fa421bc43ff710f1c76fba3e4b6223c0138ba58d`, apart from Colab title comments. The record states that later changes to the reviewed notebook are markdown-only. This review inspected that record, **not the complete archived executed notebook or separately exported report files**. [Release record][R]

The record covers **all six backbones on the built-in sample**. This notebook does not use the STANDARD/FULL controls of the previously reviewed ViT-hosted notebook. The release record's closing reference to a remaining “FULL/BYOD” gate should therefore be corrected: BYOD and resolution stress remain distinct unverified paths; SwinV2 and EVA-02 are already included in this notebook's documented default execution. Fresh runtime status, absence of manual restarts, and separate export-file digests are not independently established by the supplied record.

**Direct local evidence:** 16 grouped checks executed against manually transcribed notebook helpers/cell excerpts and controlled synthetic fixtures. These include genuine CPU linear-probe fitting, SafeTensors export/reload, real Pillow image decoding and ZIP/directory input checks. They do not include a pretrained vision backbone.

Local environment: Python 3.13.5; NumPy 2.3.5; Pillow 12.3.0; PyTorch 2.10.0+cpu; SafeTensors 0.7.0; matplotlib 3.10.8. These differ from the notebook's hosted model environment. Excerpts are not a byte-identical downloaded notebook, and helper formatting was condensed for the offline harness.

**Not verified:** Fresh Colab/T4 execution; any pretrained model prediction; updated 5-NN scores on the actual bird corpus; full BYOD model execution; measured learner understanding; full deployment-artifact compatibility with a production consumer.

## 2. Separate judgments

| Dimension | Assessment |
|---|---|
| Technical correctness | Real feature/probe workflow with positive local training controls; incorrect 5-NN decision evaluation and unsafe BYOD file-resolution behavior require correction. |
| Promise fulfillment | The six-backbone experiment is implemented and has documented execution. Resolution stress is not implemented; diagnostic outputs are not adequately surfaced. |
| Scientific/experimental validity | Training normalization and checkpoint selection passed focused separation controls. The interactive activity does not preserve its claimed validation-only presentation. Planned multi-condition test reporting is not inherently leakage. |
| Learner orientation/progression | The task diagram and system-versus-architecture boundary are useful. The core loop combines several concepts before learners see intermediate evidence. |
| Explanations/interpretation | Principal metric names appear without enough introductory interpretation. Confusion and PCA outputs exist but are not presented as promised learning views. |
| Meaningful activity | Real nested-data experiments exist, but the guided rerun instructions and actual code disagree; the resolution switch is a no-op. |
| Interaction/recovery | Default execution has historical support. ZIP retries can depend on stale files; runtime troubleshooting contains contradictory restart guidance. |
| Completion/transfer | Machine-readable outputs and real tensor reload are strengths. The reload does not validate the full serialized semantic contract; BYOD exports are less complete than the default record. |

## 3. Prioritized findings

### M1 — Major: The 5-NN evaluator scores a different tie decision

**Location:** Section 8, `knn_predict` and `classification_metrics`, cell `f370927c`; Section 10 call site, cell `6e025182`; Section 20 generic BYOD equivalent, cell `71cf319a`. [Code][N-method] [Call site][N-main] [BYOD][N-byod]

**Observed issue:** `knn_predict` selects neighbors, counts votes, and resolves a class-vote tie by summed cosine similarity followed by lower class ID. It returns that class separately from vote fractions. The caller ignores the returned `knn_pred` and computes metrics from `knn_probs.argmax(axis=1)`. The BYOD helper similarly computes `order` but returns only vote fractions. NumPy argmax picks the first maximum, so tied vote fractions are resolved by array/class position instead of the specified cosine tie-break. [Design tie rule][S] [NumPy argmax][NP]

**Local reproduction:** Five normalized training vectors had cosine similarities `[0.99, 0.98, 0.80, 0.70, 0.60]` with labels `[1, 1, 0, 0, 2]`. Class 1 and class 0 each received two votes, but class 1 had the larger summed similarity (1.97 versus 1.50). The helper returned class 1; the evaluator scored class 0 from `[0.4, 0.4, 0.2, 0, 0, 0]`. For truth class 1, the displayed accuracy was 0 rather than the returned decision's 1. A unique-majority control agreed correctly. The BYOD helper exhibited the same tie-rule loss.

**Consequence:** The reference baseline used to interpret representation quality does not consistently implement its declared decision rule. A learner could compare the probe against an incorrectly reported 5-NN baseline. The effect on the recorded 48 bird-image test cases was not measured.

**Correction:** Pass the authoritative predicted IDs separately to the metric evaluator. Retain vote fractions as scores and do not perturb them merely to encode tie decisions. Give the default and BYOD implementations the same tested decision policy.

**Acceptance check:** Cover unique majorities, 2–2–1 ties, five-way ties, equal summed similarities, and consistent row/label ordering. Recompute the six actual 5-NN results and determine whether any recorded predictions change.

### M2 — Major: BYOD ZIPs can change or mix the dataset; CSV paths are not contained

**Location:** Section 20, `load_byod_explicit`, cell `71cf319a`. [Source][N-byod]

**Observed issue:** Archive extraction writes every file to `root / Path(name).name`, flattening directories. It reuses `byod_modern_image` without a fresh staging boundary. The later CSV reader resolves `root / filename` without enforcing containment of image paths within the selected dataset.

**Local reproductions:** A fixture of 60 unique synthetic images, two literal labels `001` and `NA`, and explicit 40/10/10 splits was used.

| Case | Observed behavior |
|---|---|
| Flat directory | Accepted; labels and explicit splits preserved. |
| Equivalent flat ZIP, clean workspace | Accepted. |
| Directory containing `images/c0_00.png` and matching CSV reference | Accepted as a directory; rejected as a ZIP because extraction flattened the path. |
| ZIP with referenced `c0_00.png` and later `unreferenced/c0_00.png` | Accepted, but the second member silently replaced the image used for the first filename. |
| Complete ZIP followed by a ZIP missing one referenced image | Second ZIP accepted by reusing the prior image. The identical incomplete ZIP was rejected in a clean workspace. |
| CSV filename `../outside.png` | Accepted and read an image outside the selected dataset directory. All probe files were harmless, generated inside a temporary test directory. |

Missing-label, cross-split duplicate-image, and ZIP-member traversal controls were correctly rejected. This is not a claim that every archive guard is absent, nor evidence of code execution or network exfiltration.

**Consequence:** A corrected or second upload can be evaluated against unintended image bytes; directory and ZIP inputs do not have equivalent semantics; a CSV can include images outside the intended dataset boundary. Successful validation does not reliably establish which uploaded dataset was used.

**Correction:** Validate the entire archive before writing; preserve safe relative structure; reject duplicate normalized destinations; stage each attempt in a fresh owned directory; and commit the staged dataset only after validation. Apply resolved-path containment and symlink policy to CSV image references too. Never recursively delete arbitrary user directories.

**Acceptance check:** Directory/ZIP equivalence; nested directories; duplicate basenames and normalized paths; incomplete-then-corrected and complete-then-incomplete retries; absolute, traversal, and symlink image references; and successful explicit-split preservation. Stale files must never satisfy the current archive's manifest.

### M3 — Major: Resolution stress is advertised and selectable but does not execute

**Location:** Section 18, cell `23084f05`. [Source][N-activities]

**Observed issue:** Markdown promises degradation of held-out images to 96×96 and 160×160, model-native preprocessing, and reuse of the selected probe. The implementation initializes `resolution_stress_rows = []` and prints a message in either flag branch. It does not resize images, load any selected system, predict, evaluate, or create stress results.

**Local reproduction:** Executing the transcribed cell with the flag off and on produced zero rows in both cases. With the flag on, the message merely said this is an optional extension.

**Consequence:** A learner can enable the documented experiment and receive no experiment. This is missing functionality, not a legitimate finding that resolution had no effect. It does not block the default path because the switch is off there.

**Correction:** Implement the bounded experiment using the already-selected probe and native transform, or remove the executable-looking control and state clearly that the section is an unimplemented proposal requiring additional code.

**Acceptance check:** Enabled mode must produce per-model results for the unchanged source and the stated degradation levels, preserve probe identity, and never refit/reselect. Disabled mode must visibly skip the experiment. A proposal-only disposition must not claim that turning on a flag performs it.

### M4 — Major: The data-efficiency instructions and displayed results use different splits

**Location:** Section 15, cell `f3c4811d`, and “Try it yourself — one controlled change,” cell `guided-01`. [Experiment][N-dataeff] [Execution][N-activities] [Instructions][N-ending]

**Observed issue:** The instructions ask learners to change labeled-data availability, compare validation behavior, and keep exploration separate from a frozen canonical test result. The existing experiment calls `fit_probe` with test features, calculates test metrics for each condition, and prints test accuracies. The workflow provides no matching validation-only presentation or explicit experiment/freeze handoff. Section 10 has already displayed test scores before the learner reaches the activity instructions.

**Local reproduction:** The nested subsets correctly contained 36, 72, and 108 training rows and were nested. A test-access sentinel stopped the activity's actual argument construction when it requested test features. Source inspection also confirms that the following statements score and print the test targets.

**Consequence:** Following the recommended rerun path exposes test results during the exploration the prose asks the learner to keep separate. The learner cannot follow the stated validation-only procedure without editing implementation code.

**Important qualification:** The predeclared 6/12/18-condition test comparison is not inherently invalid. Reporting a fixed test learning curve is different from choosing new settings in response to its test scores. This finding concerns the interactive instructions and lack of a separated development path; it does not establish that the historical default run used test labels for optimizer updates or checkpoint selection.

**Correction:** Provide a validation-only activity function and a separate result namespace/export. Record the activity plan before outcomes. Either implement explicit finalization for subsequent test evaluation or avoid implying that a persisted freeze already exists. State that an already-viewed test set cannot be made newly untouched by rerunning.

**Acceptance check:** A sentinel preventing any activity read of test features, labels, metrics, or files should pass. The activity should display validation loss/accuracy, selected epoch, actual training counts, and a clearly labeled difference. Default main results and artifacts must remain unchanged.

### M5 — Major, learner-facing: The diagnostic lesson is computed but not shown

**Location:** Sections 13–16, cells `a60da6ff`, `7b82d0ea`, `cd6f5ec4`; opening roadmap and design objectives. [Main outputs][N-main] [PCA][N-activities] [Learning objectives][S]

**Observed issue:** Section 13 is titled “Per-species confusion and cross-model consensus,” but the cell displays only a count of difficulty categories. Confusion matrices and per-class values exist in data structures and later exports, not in an inline diagnostic view. The shown execution sequence does not present representative training images or a concrete misclassification/disagreement gallery. Section 16 makes six PCA figures, saves them, and calls `plt.close(fig)` without `plt.show()` or a display operation.

**Local reproduction:** The PCA lifecycle probe created a valid PNG and ended with zero open figures and no display call. This used synthetic features and an Agg backend; it is not a Colab screenshot. The code's saved-versus-displayed distinction is the basis of the finding.

**Consequence:** An image-classification learner sees model names, metric values, and category counts without being shown the images or diagnostic plots needed to connect those values to visual mistakes. Files existing on disk does not fulfill the promised guided inspection without a surfaced view or clear retrieval instructions.

**Correction:** Present training examples before modeling; display class-specific confusion/recall; show a small deterministic, correctly labeled disagreement gallery; and display PCA plots inline or provide direct notebook-accessible links. Explain that a two-dimensional projection does not establish the classifier's performance and include explained variance for context. Introduce the meaning and direction of accuracy, macro-F1, top-3 accuracy, and log-loss at first use, with a short worked contrast between accuracy and confident errors.

**Acceptance check:** A learner using only normal cell outputs can point to a species confusion, identify an actual mistaken or disputed image, and explain the corresponding metric. If no mistakes/disagreements exist, show that result explicitly rather than manufacturing examples. Default enabled PCA must visibly produce its advertised view.

### M6 — Major: Reload verifies tensors but not the complete serialized artifact contract

**Location:** Section 10, `save_probe_artifact` and `reload_probe_verify`, cell `6e025182`; generic equivalent in Section 20. [Default artifact][N-artifact] [BYOD artifact][N-byod-artifact]

**Observed issue:** The exporter writes `manifest.json`, but the verifier accepts the already-held manifest dictionary rather than reconstructing metadata from the serialized directory. It checks base identity and feature dimensions and genuinely reloads the SafeTensors file. It does not validate/use the serialized class order, or check the recorded probe-file digest before reading the tensors. The default class mapping still comes from `CLASS_KEYS`; the generic verifier receives class count externally.

**Local reproduction:** A six-class probe was actually trained on synthetic numerical features and serialized. The unchanged reload returned maximum probability difference 0.0, and a mismatched base revision was rejected. Reversing the serialized `class_order` and falsifying the manifest's probe checksum still returned 0.0, both when the caller retained the old dictionary and when it passed the changed dictionary. Interpreting the 12 score rows using the reversed class order would change all 12 semantic labels; the verifier did not examine that mapping.

**Consequence:** Numeric parity can pass while the artifact metadata a downstream consumer uses is inconsistent. This is partial fresh-boundary verification, not absence of a real tensor reload. It does not prove that a clean artifact export had incorrect labels.

**Correction:** Reconstruct from the artifact directory: read and validate the stored manifest, file identity, class order, dimensions, normalization state, and base identity. Remove reliance on training-session objects/global label lists for serving reconstruction. Compare labeled outputs as well as probabilities. Provide a small end-to-end image-to-label reload example if that is the intended consumer boundary; do not claim production artifact compatibility without testing it.

**Acceptance check:** Reload in a fresh scope with only documented artifacts, base identity, and query input. Reject changed/missing manifests, reversed labels, invalid dimensions, and wrong probe hashes. Unchanged artifacts must preserve numerical parity within the stated tolerance and identical semantic labels.

## 4. Additional findings and compliance issues

### Minor — Probability validation is incomplete

The common metric helper checks row-length equality indirectly with `zip(..., strict=True)`, which is better than silently dropping rows. It does not reject invalid probability values or row sums. A fault injection with every value equal to 2 was accepted and produced log-loss `-0.6931471805599453`. This is a robustness gap, not evidence that the trained softmax outputs had these values. Validate shape, finite values, valid class IDs, nonnegative scores, and normalized rows before scoring. [Metric helper][N-method]

### Minor, mandatory BYOD disclosure gap — Privacy and practical limits

The BYOD section lists “recommended” counts that the loader actually enforces. It should call these enforced requirements. It also needs point-of-use disclosure that images are processed in the selected hosted runtime and a warning against unauthorized sensitive/restricted uploads, as required by DAT17–18. Do not imply that local Python inside Colab means on-premises processing. Input class/row checks currently happen after image decoding; move cheap preflight checks before expensive processing. [BYOD][N-byod] [Fleet requirements][F]

### Minor — Runtime recovery text conflicts with the actual setup

Section 4 says Python 3.12, but setup uses `sys.executable` and the recorded execution used Python 3.13.15. The code intentionally retains already-loaded NumPy 2.x and records that choice; it is not an isolated per-model environment design. The setup exception says **restart the Python session without deleting the runtime**, whereas generic troubleshooting says to start a fresh runtime. Those are different actions and can repeat installation. Troubleshooting also refers to a “default tier” not provided by this notebook. Align the prose with the supported path and observed environment; do not generalize one successful record to every fresh host image. [Runtime][N-runtime] [Ending][N-ending] [Execution record][R]

### Minor — Record completeness and BYOD completion visibility

Runtime, model revisions, training/validation identities, features, and probe files are exported. The default prediction CSV retains only top-one/top-two scores, so it cannot independently reproduce full multiclass log-loss from that CSV alone. The complete fitted probe and features can reproduce it, but that requires executing reconstruction. Export all class scores when the intention is simple metric auditing.

The BYOD branch writes a summary and artifacts but not the same full per-image/per-class diagnostic inventory as the default. The terminal output-inventory check concerns only the built-in sample. Label the two result scopes clearly and add a BYOD completion inventory, query predictions, and export instructions. These are not claims that the existing BYOD branch is inference-only—it really adapts and evaluates. [Exports][N-export] [BYOD outputs][N-byod-artifact] [Ending][N-ending]

### Minor — Limitations and resource semantics need precision

The label “tutorial” does not by itself explain the effect of 48 test photographs or potential observer/pretraining overlap. State these limits explicitly. Observer identity is retained but not summarized as overlap in this notebook.

The latency helper measures warmed backbone feature-forward time, with CUDA synchronization, excluding image preprocessing, transfers, and the learned probe. Preserve the useful measurement, but name it accordingly rather than implying full image-to-label service latency. [Latency][N-artifact]

## 5. Positive findings and non-findings

- **Real experiment, not six placeholder model names:** The source loads frozen backbones sequentially, obtains finite two-dimensional pre-logit features, fits a shared probe policy, and computes held-out metrics. The historical default execution covers all six models. [Source][N-main] [Record][R]
- **Appropriate comparison scope:** Native preprocessing and pretraining differences are acknowledged as part of each pretrained system; they are not concealed confounders in a claim of architecture-only causality. [Opening][N-open]
- **Train/validation separation:** Changing only synthetic test feature values did not change training means/stds, selected epoch, or learned weights in the local control. The fitting helper never receives test labels. This does not certify the entire notebook but supports the inspected boundary. [Helper][N-method]
- **Literal BYOD label preservation:** The directory parser preserved `001` and `NA` without the CSV type-inference failure found in a different notebook. Explicit user split assignments were also preserved. No fixed `U128` label truncation was introduced in the inspected path. [BYOD][N-byod]
- **Actual integrity checks:** Built-in data and model files have expected sizes/digests; duplicate decoded images are checked; the backbone load is strict; feature dimensions and finiteness are checked. ZIP traversal-member and cross-split duplicate-image rejections passed local controls. [Data][N-data] [Model checks][N-snapshot] [Extraction][N-main]
- **Actual reload evidence:** Tensor reconstruction preserved probabilities exactly in the synthetic local control. The shortcoming is the surrounding semantic manifest boundary, not a fabricated reload success message. [Artifact][N-artifact]
- **Nested sample sizes are genuinely controlled:** The 36/72/108 subsets are nested and reused across model conditions. Their existence should not be discarded while correcting the learner's validation-only rerun route. [Experiment][N-dataeff]
- **No reason to demand artificial backbone fine-tuning:** A learned linear probe is real adaptation for the declared E2E comparison. Similarly, using test features distinct from adaptation data is meaningful inference; a separate arbitrary upload is not required merely to call it inference.

## 6. Promise-to-evidence matrix

| Promise | Evidence and judgment |
|---|---|
| Compare six pretrained systems on one task | Implemented; historical six-model run recorded. Preserve the comparison. |
| Compare 5-NN with linear probes | Implemented, but 5-NN tie handling is scored inconsistently: M1. |
| Use training statistics and validation-only selection | Supported by inspected implementation and targeted positive probe; not a full runtime certification. |
| Inspect per-species confusion and visual diagnostics | Data computed/exported; learner-facing presentation incomplete: M5. |
| Study data efficiency | Predefined experiment implemented; optional validation-only rerun instructions disagree with code: M4. |
| Enable source-resolution stress | Not implemented: M3. |
| Use explicitly split own data | Positive parser path exists; archive equivalence, source containment, and retry isolation fail: M2. Full backbone-backed execution remains unverified. |
| Prove a reusable artifact reload | Genuine tensor parity, incomplete serialized semantic boundary: M6. |
| Export enough results to retain the experiment | Substantial default exports; clearer current-scope/BYOD inventory, full score vectors, and retrieval guidance recommended. |

## 7. Specification relationship

Do not retroactively relabel the notebook as declaring 2.2. Its own declaration remains 2.1. The current 2.2 review map identifies relevant requirements, not a blanket certification.

| Finding | Relevant requirement family |
|---|---|
| M1 | EVAL2/EVAL8, UNC1, and repository specification's explicit cosine-sum 5-NN tie rule |
| M2 | DAT13/DAT19, VAL2/VAL3, input/ZIP safety requirements in §20 |
| M3 | GDL7–10 and promise-to-evidence fulfillment |
| M4 | GDL10, SPL6–7; distinguish preplanned test curves from development-time feedback |
| M5 | GDL6/GDL8/GDL9/GDL11–12, EVAL3 |
| M6 | ART2/ART5, VER2–5 |
| BYOD disclosure | DAT12, DAT17–19, VAL6 |
| Verification still required | REL execution requirements: distinguish static/local, maintainer-recorded hosted, and untested optional paths |

A `SHOULD`-level instructional concern can still be a Major learner-facing defect under the review framework. Conversely, a modest-severity issue against an applicable `MUST` is still a conformance gate. No averaged score is used.

## 8. Readiness and correction sequence

**Needs revision for the full advertised learning experience.** Retain the existing six-model run as scoped historical evidence.

First, repair 5-NN decision evaluation and BYOD source isolation; these affect what data/method is actually scored. Next, implement or honestly withdraw resolution stress, provide a validation-only activity path, and surface the diagnostic results. Complete manifest-driven artifact reconstruction and align runtime/privacy/export instructions.

After correction, record an exact-revision hosted run of the default six-model workflow. Recompute actual bird-image 5-NN predictions under the intended tie policy; keep old results labeled as historical rather than silently overwriting the evidence. Qualify representative directory/ZIP BYOD inputs through real feature extraction, adaptation, test inference, artifact reconstruction, and export. Add negative archive/path/label cases, semantic manifest mutations, and the separate activity/stress paths.

An observed learner walkthrough should establish whether participants can explain what was frozen versus trained, read the baseline and loss metrics, inspect an actual mistake, make a controlled change without test feedback, recover from an invalid upload, and retain the correct output record. This review identifies likely barriers; it does not claim measured learning effectiveness.

**Repository changes made during this review: none.**

## 9. Offline probe inventory

The companion ZIP contains `notebook_excerpts.py`, `run_probes.py`, `results/probe_results.json`, `results/run_log.txt`, and a synthetic PCA lifecycle image. It contains no bird photos, pretrained checkpoints, or authentication material.

The JSON reports 16 grouped checks, including positive controls. Successful execution of this harness means the reported local behaviors were reproduced—not that the source notebook passed acceptance. Invalid-output and artifact tests intentionally inject faults. All archive/path fixtures are generated and destroyed within a temporary test directory.

## Sources

[N-open]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1-L178
[N-runtime]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L179-L285
[N-data]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L325-L480
[N-snapshot]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L530-L605
[N-method]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L605-L750
[N-main]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L775-L1158
[N-artifact]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L844-L925
[N-dataeff]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1158-L1235
[N-activities]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1200-L1390
[N-export]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1380-L1480
[N-byod]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1490-L1680
[N-byod-artifact]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1680-L1820
[N-ending]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/tutorials/DIMER_Modern_Image_Classification_Workshop.ipynb#L1810-L2070
[R]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/docs/release-verification.md
[S]: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/cf3dbadaade69dd320b6018832646601a2cd4df9/docs/modern-image-classification-workshop-spec.md
[F]: https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md
[NP]: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
