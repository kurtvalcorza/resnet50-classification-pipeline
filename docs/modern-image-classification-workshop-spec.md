# DIMER Modern Image Classification Workshop

## From ResNet to MobileNet, ConvNeXt and Vision Transformers

**Proposed filename:** `DIMER_Modern_Image_Classification_Workshop.ipynb`  
**DIMER Notebook Specification:** `2.1`  
**Profile:** `E2E`  
**Pedagogical mode:** `WORKSHOP`  
**Comparison scope:** `MULTI-MODEL`  
**Standalone:** `true`  
**Recommended runtime:** Kaggle / Colab Tesla T4  
**Task:** Single-label image classification  
**Canonical dataset:** Six-species CC0 iNaturalist bird dataset  
**Canonical adaptation:** Frozen backbone + identical linear probe  
**Backbone fine-tuning:** None  
**Classes:** 6  
**Default split:** 108 train / 24 validation / 48 test

---

# 1. Purpose

This workshop compares six generations and design philosophies of modern image classifiers:

1. **ResNet-50**
2. **MobileNetV4-Conv-Small**
3. **ConvNeXt-Tiny**
4. **ViT-B/16**
5. **SwinV2-Tiny**
6. **EVA-02 Base 448**

The objective is not to compare their native ImageNet-1k heads.

Instead, each pretrained model is used as a **frozen feature extractor**:

```text
image
  ↓
pretrained backbone
  ↓
pooled pre-logit representation
  ↓
identical 6-class linear probe
  ↓
bird species prediction
```

This creates a common downstream task and a common evaluation contract.

---

# 2. Why not compare the native ImageNet heads?

All six checkpoints end in ImageNet-1k classification heads, but the tutorial dataset contains six bird species whose relationship to ImageNet labels is not clean or uniform.

Directly comparing:

```text
ImageNet head → six custom classes
```

would require arbitrary label mappings.

That would confound:

- model quality;
- label ontology;
- ImageNet vocabulary coverage; and
- mapping policy.

Instead, this workshop asks:

> **How useful is each frozen pretrained representation when exactly the same lightweight classifier is fitted on top of it?**

---

# 3. Why frozen linear probes?

A linear probe provides a controlled representation test.

For every model:

```text
backbone weights = frozen
classifier = newly fitted linear layer
training set = identical
validation set = identical
test set = identical
optimizer = identical
number of epochs = identical
metric implementation = identical
```

Only the underlying pretrained representation changes.

This is substantially cleaner than comparing six model-specific full fine-tuning recipes.

---

# 4. Learning objectives

By the end of the workshop, the learner should be able to:

1. explain the progression from CNNs to modern ConvNets and vision transformers;
2. distinguish a pretrained classifier head from a reusable visual representation;
3. extract pre-logit features from different `timm` architectures;
4. describe feature dimensionality and pooling semantics;
5. fit a common linear probe without changing the backbone;
6. compare majority and 5-NN baselines with a trained probe;
7. evaluate accuracy, macro F1, top-3 accuracy and log loss;
8. inspect per-species confusion;
9. compare sample efficiency of different pretrained representations;
10. interpret resource use alongside predictive quality;
11. understand how input resolution and pretraining recipes affect the comparison; and
12. apply the same probe protocol to a labelled image dataset of their own.

---

# 5. Model set

## 5.1 ResNet-50

| Field | Value |
|---|---|
| Model | `timm/resnet50.a1_in1k` |
| Revision | `767268603ca0cb0bfe326fa87277f19c419566ef` |
| Architecture family | Residual CNN |
| License | Apache-2.0 |
| Native input | 224×224 |
| Weight bytes | 102,469,840 |
| Weight SHA-256 | `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb` |

Conceptual path:

```text
convolutions
→ residual blocks
→ global pooling
→ representation
```

---

# 5.2 MobileNetV4-Conv-Small

| Field | Value |
|---|---|
| Model | `timm/mobilenetv4_conv_small.e2400_r224_in1k` |
| Revision | `331fb803779522b685cf942e15f914fb6741c1eb` |
| Architecture family | Efficient/mobile CNN |
| License | Apache-2.0 |
| Native input | 224×224 |
| Approx. parameters | 3.8M |
| Weight bytes | 15,223,016 |
| Weight SHA-256 | `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4` |

This is the efficiency reference.

---

# 5.3 ConvNeXt-Tiny

| Field | Value |
|---|---|
| Model | `timm/convnext_tiny.in12k_ft_in1k` |
| Revision | `aa096f03029c7f0ec052013f64c819b34f8ad790` |
| Architecture family | Modern ConvNet |
| License | Apache-2.0 |
| Native input | 224×224 |
| Weight bytes | 114,374,272 |
| Weight SHA-256 | `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95` |

ConvNeXt represents the redesign of a conventional convolutional architecture using design ideas popularized by vision transformers.

---

# 5.4 ViT-B/16

| Field | Value |
|---|---|
| Model | `timm/vit_base_patch16_224.orig_in21k_ft_in1k` |
| Revision | `e0bd370de6799e8d1f47a911174ff4c3708e2323` |
| Architecture family | Global Vision Transformer |
| License | Apache-2.0 |
| Native input | 224×224 |
| Patch size | 16×16 |
| Parameters | ~86.6M |
| Weight bytes | 346,284,714 |
| Weight SHA-256 | `669b949ea91fd19217f200cee259780bde32210c1eb9a5af3859f0dd8346b2ec` |

Conceptual path:

```text
image
→ fixed patches
→ token sequence
→ global transformer blocks
→ pooled representation
```

---

# 5.5 SwinV2-Tiny

Use only **Tiny** in the workshop.

Do not include the sibling Small checkpoint in the canonical comparison.

| Field | Value |
|---|---|
| Model | `timm/swinv2_tiny_window8_256.ms_in1k` |
| Revision | `650d02aabf05e8adbd060a739ab39e39f53da639` |
| Architecture family | Hierarchical windowed transformer |
| License | MIT |
| Native input | 256×256 |
| Window size | 8 |
| Weight bytes | 114,918,618 |
| Weight SHA-256 | `c47f52b4556ff4436aa9502f5efbc93aac77fdab42d9d70dd845931757ff5d65` |

Conceptually:

```text
patches
→ local window attention
→ shifted windows
→ hierarchical patch merging
→ pooled representation
```

### Why Tiny rather than Tiny + Small

The workshop is comparing architectural families rather than scaling curves.

Including both Swin variants would overweight one family and increase runtime without adding a new architectural idea.

---

# 5.6 EVA-02 Base 448

| Field | Value |
|---|---|
| Model | `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` |
| Revision | `81063ecfe9c381a16a19d06f396d6c7011aa426a` |
| Architecture family | Large modern vision transformer |
| License | MIT |
| Native input | 448×448 |
| Patch size | 14×14 |
| Parameters | ~87.1M |
| Weight bytes | 348,492,484 |
| Weight SHA-256 | `533937d6f9f8f8d4f50627ef0d00829a5015861a5b67e1c69c5a5e45b7dc2609` |

EVA-02 also differs in:

- much higher input resolution;
- masked-image-modeling pretraining;
- ImageNet-22k lineage; and
- materially higher compute.

It serves as the high-compute reference.

---

# 6. Models intentionally excluded

Do not include:

- SigLIP / SigLIP 2;
- BioCLIP;
- PubMedCLIP;
- zero-shot vision-language classifiers;
- planned EfficientNet / DeiT / MaxViT / CvT models.

Reason:

The core notebook should compare **live closed-set visual backbones** under one supervised feature-transfer protocol.

Vision-language models deserve their own zero-shot / retrieval comparison.

Planned models can join this workshop once they are live and clean-runtime qualified.

---

# 7. Important interpretation boundary

This is **not a pure architecture ablation**.

The checkpoints differ in:

- architecture;
- input resolution;
- training recipe;
- pretraining dataset;
- pretraining objective;
- augmentation;
- intermediate fine-tuning;
- model size.

Therefore results describe the **complete pretrained checkpoint systems**, not architecture in isolation.

The notebook must state this prominently.

---

# 8. Dataset

Reuse the existing DIMER ViT dataset:

**iNaturalist CC0 bird photographs — six species**

Properties:

```text
180 photographs
6 species
30 photographs per species
one observer per selected photo/species where practical
CC0 1.0
~19.2 MB total
```

Every image is pinned by:

```text
DIMER record ID
species
iNaturalist photo ID
observation ID
observer
byte length
SHA-256
extension
```

No photographs are redistributed by the repository.

They are fetched from the iNaturalist open-data bucket at runtime.

---

# 9. Classes

Canonical ordering:

```text
0  song_sparrow
1  chipping_sparrow
2  white_throated_sparrow
3  dark_eyed_junco
4  house_finch
5  american_goldfinch
```

Display names:

```text
Song Sparrow
Chipping Sparrow
White-throated Sparrow
Dark-eyed Junco
House Finch
American Goldfinch
```

Scientific names should also be retained in provenance.

---

# 10. Dataset split

Reuse the existing deterministic ViT carrier split:

```text
per species:

18 train
4 validation
8 test
```

Total:

```text
train        108
validation    24
test          48
```

Seed:

```text
42
```

The exact existing 180-row DIMER photo manifest should be carried into the standalone workshop.

---

# 11. Data integrity

Before model loading:

- fetch all 180 selected photographs;
- verify byte size;
- verify SHA-256;
- decode with Pillow;
- convert to RGB;
- compute decoded-pixel digest;
- reject duplicate decoded images;
- verify exactly 30 records/species;
- reproduce 18/4/8 split;
- verify split disjointness.

Export:

```text
dataset_manifest.json
```

---

# 12. Why this dataset is appropriate

The six species are visually similar enough to make the classification task non-trivial.

The dataset also tests transfer beyond the checkpoint's original 1000-class head.

The workshop therefore asks:

> How linearly separable are these six species in each pretrained representation?

It does **not** ask:

> Which model has the best native ImageNet classifier?

---

# 13. Notebook profile

Because every model receives a fitted downstream classifier, this is an:

```text
E2E
```

notebook.

Pedagogical mode:

```text
WORKSHOP
```

The adaptation is intentionally limited to:

```text
frozen backbone
+
new linear classifier
```

No backbone gradients occur.

---

# 14. Canonical workflow

```text
install
→ verify six model snapshots
→ fetch/verify dataset
→ validate and split
→ majority baseline
→ for each model:
      load backbone
      native preprocessing
      extract frozen train/val/test features
      representation baseline
      fit common linear probe
      select probe epoch on validation loss
      test evaluation
      export probe artifact
      fresh probe reload verification
      record latency/memory
      unload backbone
→ cross-model comparison
→ data-efficiency experiment
→ error/confusion analysis
→ representation geometry
→ resource tradeoffs
→ BYOD
→ interpretation
```

Models MUST be executed sequentially to limit GPU memory.

---

# 15. Runtime

Use:

```text
Python 3.12
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
timm==1.0.29
safetensors==0.8.0
numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
```

Recommended:

```text
Tesla T4 or better
```

CPU support MAY remain technically functional for smaller models, but EVA-02 makes CPU execution impractical for the complete workshop.

---

# 16. Model acquisition

Carry the exact DIMER snapshot manifest for every model.

For every snapshot:

```text
model ID
immutable revision
license
config.json
model.safetensors
per-file byte size
per-file SHA-256
```

Acquisition MUST:

1. fetch only manifest-listed files;
2. use the immutable revision;
3. verify every byte size;
4. verify every digest;
5. reject drift;
6. load from the verified local snapshot.

No mutable branch may define model identity.

---

# 17. Standalone implementation

The notebook MUST NOT:

```text
git clone
pip install -e .
import the six DIMER pipeline packages
fetch DIMER source code
call DIMER services
```

Use `timm` directly as the common general-purpose model library.

The notebook carries:

- model registry;
- manifests;
- snapshot verifier;
- dataset manifest;
- image validation;
- feature extraction helpers;
- probe trainer;
- metrics;
- artifact exporter;
- artifact verifier;
- BYOD loader.

---

# 18. Native preprocessing

Every model must use its own pinned `timm` preprocessing configuration.

Use:

```text
resolve_model_data_config(model)
create_transform(..., is_training=False)
```

or the equivalent pinned `timm` API.

Do not force every model to 224×224.

This means:

```text
ResNet-50       224
MobileNetV4     224
ConvNeXt-Tiny   224
ViT-B/16        224
SwinV2-Tiny     256
EVA-02 Base     448
```

The input-resolution difference is part of the deployed checkpoint system.

---

# 19. Feature extraction contract

For every model:

```python
with torch.inference_mode():
    features = model.forward_features(batch)
    pre_logits = model.forward_head(features, pre_logits=True)
```

The notebook MUST validate:

```text
pre_logits.ndim == 2
pre_logits.shape[0] == batch_size
pre_logits.shape[1] == model.head_hidden_size   # timm pre-logit width; MobileNetV4 differs from num_features
all values finite
```

Output unit:

```text
one pooled representation per image
```

These representations are not predictions.

---

# 20. Feature export semantics

For each model record:

```text
model
image_id
split
label
feature_dim
```

The full feature matrix does not need to be retained as a CSV.

Recommended:

```text
features/<model_key>.npz
```

containing:

```text
image_ids
labels
splits
features
```

Identifiers MUST remain aligned with vectors.

---

# 21. Frozen-backbone invariant

During feature extraction:

```text
model.eval()
requires_grad = False
torch.inference_mode()
```

Assert:

```text
trainable backbone parameters == 0
```

No optimizer may receive a backbone parameter.

---

# 22. Common feature preprocessing

Each model's train representations may have a different dimension and numeric scale.

Before probe training:

1. calculate feature mean using **training features only**;
2. calculate feature standard deviation using **training features only**;
3. replace extremely small standard deviations with `1e-6`;
4. standardize train/validation/test using those training statistics.

Do not calculate normalization statistics on validation or test features.

---

# 23. Majority baseline

The dataset is balanced.

Majority accuracy:

```text
1 / 6 = 16.67%
```

Use deterministic class-ID tie-breaking.

This baseline is shared across all six models.

---

# 24. Representation-only 5-NN baseline

Before fitting a probe, score each model using a 5-nearest-neighbor classifier over its frozen features.

Protocol:

1. standardize from train statistics;
2. L2-normalize feature vectors;
3. cosine similarity;
4. choose five most similar training vectors;
5. majority vote;
6. tie-break by:
   - higher summed cosine similarity;
   - then lower class ID.

Report:

```text
5-NN accuracy
5-NN macro F1
```

This measures representation usefulness without learning a parametric classification head.

---

# 25. Common linear probe

For each model:

```text
input dimension = model.head_hidden_size
output dimension = 6
```

Architecture:

```text
nn.Linear(feature_dim, 6)
```

Nothing else.

No hidden layers.

---

# 26. Probe initialization

Use deterministic zero initialization:

```text
weight = 0
bias = 0
```

This avoids differences caused by random probe initialization.

---

# 27. Probe training

Canonical configuration:

```text
optimizer = AdamW
learning_rate = 0.05
weight_decay = 1e-4
epochs = 200
batching = full training feature matrix
loss = cross entropy
device = CPU
```

Training the probes on CPU:

- removes probe-training GPU differences;
- is computationally trivial;
- leaves GPU timing focused on backbone inference.

No shuffle is required for full-batch optimization.

---

# 28. Probe model selection

After every epoch:

```text
validation cross-entropy
validation accuracy
```

Selection rule:

```text
lowest validation cross-entropy
```

Tie-break:

```text
earliest epoch
```

The test set MUST NOT influence selection.

Restore the selected epoch before held-out evaluation.

Record two flags with every probe: `selected_at_first_epoch` (the probe barely trained; with a zero-initialised
layer and AdamW this means the learning rate is too high) and `selected_at_epoch_cap` (validation loss was still
falling at the last epoch, which is expected when the validation images are already separable). A local CPU run
on 2026-09-26 found that lr 0.05 selected epoch 1 for five of the six backbones; lr 0.001 over 1000 epochs
selected epochs 34–46 for ResNet-50, MobileNetV4 and SwinV2 and the cap for ConvNeXt, ViT and EVA-02.

---

# 29. Required metrics

For each model report:

### Primary

- test accuracy
- test macro F1

### Complementary

- top-3 accuracy
- log loss
- per-class precision
- per-class recall
- per-class F1
- confusion matrix

### Baselines

- majority
- 5-NN frozen-feature baseline

---

# 30. Comparison table

Produce:

| Model | Feature dim | 5-NN acc | Probe acc | Macro F1 | Top-3 | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | measured | ... | ... | ... | ... | ... |
| MobileNetV4 | measured | ... | ... | ... | ... | ... |
| ConvNeXt-Tiny | measured | ... | ... | ... | ... | ... |
| ViT-B/16 | measured | ... | ... | ... | ... | ... |
| SwinV2-Tiny | measured | ... | ... | ... | ... | ... |
| EVA-02 Base | measured | ... | ... | ... | ... | ... |

Do not hard-code quality expectations.

---

# 31. Per-species comparison

Produce one common table:

| Species | ResNet | MobileNetV4 | ConvNeXt | ViT | SwinV2 | EVA-02 |
|---|---:|---:|---:|---:|---:|---:|
| Song Sparrow | recall | ... | ... | ... | ... | ... |
| Chipping Sparrow | recall | ... | ... | ... | ... | ... |
| White-throated Sparrow | ... | ... | ... | ... | ... | ... |
| Dark-eyed Junco | ... | ... | ... | ... | ... | ... |
| House Finch | ... | ... | ... | ... | ... | ... |
| American Goldfinch | ... | ... | ... | ... | ... | ... |

Macro metrics alone must not hide species-specific failure.

---

# 32. Common confusion analysis

For every model:

- calculate six-way confusion matrix;
- identify strongest off-diagonal pair;
- count correct / incorrect predictions;
- surface highest-confidence errors.

Then create a cross-model disagreement table:

```text
image_id
gold_label

resnet_prediction
mobilenet_prediction
convnext_prediction
vit_prediction
swin_prediction
eva_prediction

n_models_correct
```

---

# 33. Consensus analysis

For every test image calculate:

```text
number of models predicting gold label
number of unique predicted labels
```

Useful categories:

### unanimous correct

All six correct.

### majority correct

4–5 models correct.

### split

2–3 correct.

### shared hard case

0–1 correct.

This is descriptive error analysis.

It must not be presented as an ensemble deployment recommendation.

---

# 34. Representation geometry

For each model, use L2-normalized **raw pre-logit features**.

On held-out features compute:

```text
mean cosine similarity:
  same class

mean cosine similarity:
  different class
```

Define:

```text
separability_gap =
mean_same_class_cosine
-
mean_different_class_cosine
```

Report it as a diagnostic representation statistic.

It is not an accuracy metric.

---

# 35. Optional train-fitted PCA visualization

Recommended default:

```text
RUN_PCA_VISUALIZATION = True
```

For each model:

1. fit two-component PCA using **training features only**;
2. transform test features;
3. display test samples by species.

PCA is explanatory visualization only.

No model selection may depend on the projection.

---

# 36. Data-efficiency experiment

This should be a major workshop section.

The feature extraction is already complete, so fitting additional linear probes is inexpensive.

Use nested training subsets:

```text
6 images/species   = 36 images
12 images/species  = 72 images
18 images/species  = 108 images
```

Selection:

- deterministic;
- from training split only;
- nested;
- same images for every model.

Validation and test sets remain unchanged.

---

# 37. Data-efficiency outputs

For every model/fraction:

```text
train_images_per_class
train_images_total
best_epoch
validation_loss
test_accuracy
test_macro_f1
```

Produce:

| Model | 6/class | 12/class | 18/class |
|---|---:|---:|---:|
| ResNet-50 | acc | acc | acc |
| MobileNetV4 | ... | ... | ... |
| ConvNeXt | ... | ... | ... |
| ViT | ... | ... | ... |
| SwinV2 | ... | ... | ... |
| EVA-02 | ... | ... | ... |

This teaches **representation sample efficiency**.

---

# 38. Workshop exercise

Before showing the full-data results, ask the learner to predict:

1. Which model will have the strongest frozen representation?
2. Which will use the least memory?
3. Which will extract features fastest?
4. Will the largest model necessarily produce the highest probe accuracy?
5. Which architectures might perform best with only six training images per species?

Then reveal the measured results.

No expected answers are encoded.

---

# 39. Resource measurement

For each backbone record separately:

```text
snapshot verification seconds
model load seconds

parameter count
weight bytes
feature dimension
native input resolution

single-image inference latency
bulk feature-extraction time
peak GPU memory
```

Probe-training time should be measured separately.

---

# 40. Latency protocol

For comparable model inference latency:

1. load model;
2. warm up;
3. use one fixed evaluation image;
4. run several repeated single-image passes;
5. synchronize CUDA before/after each pass;
6. report median latency.

Recommended:

```text
LATENCY_REPEATS = 10
```

Do not include preprocessing download/model load time.

Also report full-dataset feature-extraction wall time separately.

---

# 41. Resource-quality table

Produce:

| Model | Params | Weights | Input | Feature dim | Probe acc | Median latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV4 | ... | 15 MB | 224 | ... | ... | ... | ... |
| ResNet-50 | ... | 102 MB | 224 | ... | ... | ... | ... |
| ConvNeXt-Tiny | ... | 114 MB | 224 | ... | ... | ... | ... |
| SwinV2-Tiny | ... | 115 MB | 256 | ... | ... | ... | ... |
| ViT-B/16 | ... | 346 MB | 224 | ... | ... | ... | ... |
| EVA-02 Base | ... | 348 MB | 448 | ... | ... | ... | ... |

Numbers are measured from the current run where appropriate.

---

# 42. Pareto analysis

Optionally identify models that are non-dominated under:

```text
higher probe accuracy
lower median latency
```

and separately:

```text
higher probe accuracy
smaller weight footprint
```

The notebook should not select one universal "best" model.

Instead explain:

> Different models occupy different accuracy/resource tradeoffs.

---

# 43. Native input resolution matters

The workshop must explicitly state:

```text
ResNet / MobileNet / ConvNeXt / ViT = 224
SwinV2 = 256
EVA-02 = 448
```

Therefore EVA-02 sees materially more pixels.

Its performance and latency reflect that advantage/cost.

The comparison is between deployable checkpoint configurations, not equal-resolution architecture experiments.

---

# 44. Pretraining matters

Checkpoint histories also differ:

- ResNet-50 — ImageNet-1k
- MobileNetV4 — ImageNet-1k recipe
- ConvNeXt-Tiny — ImageNet-12k → ImageNet-1k
- ViT-B/16 — ImageNet-21k → ImageNet-1k
- SwinV2-Tiny — ImageNet-1k
- EVA-02 — masked-image pretraining / ImageNet-22k lineage → ImageNet-1k

Consequently:

```text
probe performance ≠ architecture effect alone
```

The notebook should repeatedly use the phrase:

**pretrained checkpoint comparison**

rather than:

**architecture benchmark**

---

# 45. Linear-probe artifacts

For each model export:

```text
probes/<model_key>/
  probe.safetensors
  manifest.json
```

`probe.safetensors` contains:

```text
classifier.weight
classifier.bias
feature_mean
feature_std
```

---

# 46. Probe artifact manifest

Each manifest records:

```text
format
format_version

base_model_id
base_model_revision
base_weight_sha256

feature_interface = pre_logits
feature_dim

class_order

probe:
  best_epoch
  optimizer
  learning_rate
  weight_decay

training_sample_digest
validation_sample_digest

probe_file:
  bytes
  SHA-256
```

This is a workshop probe artifact, not a replacement for the base model.

---

# 47. Fresh reload verification

Before unloading each backbone:

1. write probe artifact;
2. construct a new `nn.Linear`;
3. load the SafeTensors probe;
4. restore feature normalization;
5. classify fixed cached test representations;
6. compare scores with the in-memory selected probe.

Required tolerance:

```text
max_abs_probability_diff <= 1e-6
```

All predicted class IDs must match exactly.

---

# 48. Base-model binding

A probe MUST refuse to load against a different backbone identity.

For example:

```text
ResNet probe + ConvNeXt backbone
```

must fail before inference.

Identity check must include at least:

```text
model ID
revision
weight SHA-256
feature dimension
```

---

# 49. Optional resolution-stress experiment

Default:

```text
RUN_RESOLUTION_STRESS = False
```

If enabled:

1. downsample test source images to a small intermediate resolution;
2. re-upsample;
3. run through the normal model-specific transform;
4. reuse the existing full-data probe.

Suggested intermediate sizes:

```text
96 px
160 px
```

This measures image-detail sensitivity without changing model architecture.

It is exploratory and must not influence probe selection.

---

# 50. Required machine-readable outputs

Write under:

```text
outputs/modern_image_classification/
```

---

# 51. `model_metrics.csv`

```text
model
architecture_family
feature_dim

knn5_accuracy
knn5_macro_f1

probe_accuracy
probe_macro_f1
probe_top3_accuracy
probe_log_loss

best_epoch
validation_log_loss
```

---

# 52. `class_metrics.csv`

```text
model
class_id
class_label
support
precision
recall
f1
```

---

# 53. `predictions.csv`

```text
image_id
gold_label

model
predicted_label
correct

top1_score
top2_label
top2_score
margin
```

---

# 54. `consensus.csv`

```text
image_id
gold_label
models_correct
unique_predictions
difficulty_category
```

---

# 55. `data_efficiency.csv`

```text
model
train_images_per_class
train_images
best_epoch
validation_loss
test_accuracy
test_macro_f1
```

---

# 56. `representation_geometry.csv`

```text
model
feature_dim
mean_same_class_cosine
mean_different_class_cosine
separability_gap
```

---

# 57. `resource_metrics.csv`

```text
model
model_id
revision

parameter_count
weight_bytes
native_input_size
feature_dim

load_seconds
median_single_image_latency_s
feature_extraction_seconds
probe_training_seconds
peak_gpu_memory_bytes
```

---

# 58. `features/*.npz`

For each model:

```text
image_ids
labels
splits
features
```

Use float32.

The feature archive is an analysis output, not a model artifact.

---

# 59. `metrics.json`

Contains all nested metrics:

```text
majority baseline
5-NN results
probe results
per-class metrics
confusion matrices
data-efficiency results
representation geometry
resource observations
reload verification
```

---

# 60. `provenance.json`

Record:

```text
notebook_spec
profile
pedagogical_mode

dataset:
  corpus identity
  license
  180 image manifest
  split seed
  split sizes
  sample digest

for every model:
  model key
  model ID
  immutable revision
  license
  weight bytes
  weight SHA-256
  config digest
  native preprocessing
  parameter count
  feature dimension

probe protocol
runtime versions
device
timings
```

---

# 61. BYOD

Because this is an `E2E` workflow, BYOD must support the **same adaptation path**, not inference only.

Recommended format:

```text
dataset.zip

train/
  class_a/
  class_b/

validation/
  class_a/
  class_b/

test/
  class_a/
  class_b/
```

or equivalent:

```text
images/
labels.csv
```

where `labels.csv` explicitly includes:

```text
filename,label,split
```

---

# 62. BYOD requirements

Recommended ceilings:

```text
2..20 classes
60..600 images total
>=10 train images per class
>=2 validation images per class
>=2 test images per class
```

Supported images:

```text
JPEG
PNG
WebP
TIFF
BMP
```

Every sample must belong to exactly one class.

No multi-label task.

---

# 63. BYOD split ownership

The notebook MUST NOT reinterpret test data as validation data.

User-supplied splits must remain:

```text
train
validation
test
```

If any required split is absent, reject clearly.

No automatic test-set reuse.

---

# 64. BYOD workflow

User data must pass through:

```text
load
→ validate
→ split-integrity check
→ six model feature extraction
→ majority baseline
→ 5-NN
→ linear probe
→ validation selection
→ test evaluation
→ probe artifact export
→ reload verification
```

This satisfies the E2E BYOD requirement.

---

# 65. BYOD practical model selection

For large BYOD sets, processing all six models may be expensive.

The optional BYOD branch MAY expose:

```text
BYOD_MODEL_KEYS
```

defaulting to all six.

Any reduced selection must be explicit and must not alter the canonical built-in workflow.

---

# 66. BYOD privacy

User images remain inside the selected notebook runtime.

Do not upload:

- confidential imagery;
- restricted datasets;
- personal imagery without authorization;
- regulated imagery

to hosted runtimes unless permitted.

Model-weight acquisition does not require transmitting user images to model hosts.

---

# 67. Default parameters

```python
USE_BYOD = False

FEATURE_BATCH_SIZE = 8
LATENCY_REPEATS = 10

PROBE_EPOCHS = 1000
PROBE_LR = 0.001   # lr 0.05 selected epoch 1 (a one-step probe) for 5 of 6 backbones
PROBE_WEIGHT_DECAY = 1e-4

KNN_K = 5

RUN_DATA_EFFICIENCY = True
DATA_EFFICIENCY_PER_CLASS = [6, 12, 18]

RUN_PCA_VISUALIZATION = True
RUN_RESOLUTION_STRESS = False

SAMPLE_SEED = 42

OUTPUT_DIR = "outputs/modern_image_classification"
```

---

# 68. Notebook cell plan

| # | Type | Section |
|---:|---|---|
| 0 | Markdown | Title, profile, objectives |
| 1 | Markdown | Why not compare ImageNet heads |
| 2 | Markdown | CNN → modern ConvNet → transformer progression |
| 3 | Code | Form parameters |
| 4 | Markdown | Runtime contract |
| 5 | Code | Install/check pinned dependencies |
| 6 | Markdown | Dataset provenance |
| 7 | Code | Fetch + digest-check 180 photographs |
| 8 | Markdown | Dataset split |
| 9 | Code | Reproduce 108/24/48 split |
| 10 | Markdown | Model registry |
| 11 | Code | Six manifests + verification helpers |
| 12 | Markdown | Linear-probe methodology |
| 13 | Code | Common metrics + probe helpers |
| 14 | Markdown | Majority baseline |
| 15 | Code | Majority floor |
| 16 | Markdown | ResNet-50 |
| 17 | Code | Load → features → kNN → probe → artifact → reload |
| 18 | Markdown | MobileNetV4 |
| 19 | Code | Same workflow |
| 20 | Markdown | ConvNeXt-Tiny |
| 21 | Code | Same workflow |
| 22 | Markdown | ViT-B/16 |
| 23 | Code | Same workflow |
| 24 | Markdown | SwinV2-Tiny |
| 25 | Code | Same workflow |
| 26 | Markdown | EVA-02 Base |
| 27 | Code | Same workflow |
| 28 | Markdown | Main comparison |
| 29 | Code | Metrics/resource tables |
| 30 | Markdown | Workshop prediction exercise |
| 31 | Code | Reveal cross-model comparison |
| 32 | Markdown | Per-species confusion |
| 33 | Code | Confusion + consensus analysis |
| 34 | Markdown | Representation geometry |
| 35 | Code | Same/different-class cosine statistics |
| 36 | Markdown | Data efficiency |
| 37 | Code | 6/12/18-per-class probe runs |
| 38 | Markdown | PCA representation views |
| 39 | Code | Train-fitted PCA projections |
| 40 | Markdown | Accuracy/resource tradeoffs |
| 41 | Code | Latency/size/Pareto analysis |
| 42 | Markdown | Optional resolution stress |
| 43 | Code | Default no-op |
| 44 | Markdown | Machine-readable exports |
| 45 | Code | CSV/NPZ/JSON/provenance |
| 46 | Markdown | BYOD |
| 47 | Code | Optional BYOD E2E branch |
| 48 | Markdown | Interpretation and limitations |
| 49 | Code | Terminal summary + output assertions |

Every executable cell must be preceded by explanatory markdown.

---

# 69. Required assertions

Dataset:

```text
180 images
30 per species
108 train
24 validation
48 test
no duplicated decoded pixels
all source digests valid
```

Each model:

```text
manifest model ID matches
revision matches
weight digest matches

all backbone parameters frozen
model.eval()

feature matrix finite
feature rows align with image IDs
feature_dim == model.head_hidden_size
```

Probe:

```text
6 output classes
class ordering identical
normalization fitted on train only
no test data in selection

best epoch selected on validation CE
probabilities finite
probabilities sum to ~1
```

Artifact:

```text
base identity matches
feature dimension matches
reload probabilities within tolerance
reload class IDs identical
```

Exports:

```text
all required outputs exist
```

---

# 70. Quality results MUST NOT be asserted

Never release-gate on claims such as:

```text
EVA-02 must win
MobileNet must be fastest
ViT must beat ResNet
ConvNeXt must outperform ResNet
larger models must be more sample-efficient
```

Those are measured outcomes.

Only structural and integrity properties are release invariants.

---

# 71. Interpretation requirements

## This is a representation-transfer comparison

The linear head is intentionally simple.

If two models differ substantially, the difference is evidence that their pretrained representations differ in usefulness for this downstream task.

---

## It is not architecture alone

Checkpoint behavior also reflects:

```text
pretraining data
training recipe
input resolution
pretraining objective
augmentation
model capacity
```

---

## Larger does not imply better

Parameter count and checkpoint size are independent dimensions from test accuracy.

---

## Native resolution is part of the cost

EVA-02 receives 448×448 inputs while several competitors receive 224×224.

That materially changes compute and available image detail.

---

## 5-NN and the linear probe answer different questions

5-NN asks:

> Are nearby frozen representations already grouped by class?

The linear probe asks:

> Can a linear decision boundary separate the classes efficiently?

---

## Data-efficiency curves matter

A representation that performs strongly with only six labelled examples per class may be valuable even if another model has slightly higher full-data accuracy.

---

## Scores are not calibrated probabilities

Probe softmax outputs are ranking/decision scores.

No universal confidence threshold is established.

---

# 72. Explicit non-goals

This notebook does not perform:

- full backbone fine-tuning;
- ImageNet benchmark reproduction;
- native-head label mapping;
- zero-shot text classification;
- multi-label classification;
- object detection;
- image retrieval;
- segmentation;
- prompt engineering;
- model ensembling;
- automatic hyperparameter search;
- test-set model selection;
- DIMER service calls.

---

# 73. Recommended repository placement

This is a cross-model workshop, so use one canonical copy:

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        modern-image-classification/
          DIMER_Modern_Image_Classification_Workshop.ipynb
          README.md
          docs/
            release-verification.md
          tools/
            build_notebook.py
            validate_notebook.py
```

The six carrier repositories can link to this notebook.

---

# 74. Release verification

Preferred environment:

```text
Kaggle Tesla T4
Python 3.12
```

Cold-start qualification must verify:

- no repository checkout;
- no DIMER service;
- empty model cache;
- empty image cache;
- automatic model/data acquisition;
- all snapshot digests;
- all 180 image digests;
- all six feature extraction paths;
- all six 5-NN baselines;
- all six linear probes;
- all six probe exports;
- all six reload checks;
- data-efficiency experiment;
- machine-readable exports.

Record:

```text
notebook commit
notebook blob

runtime
GPU

dataset digest
split sizes

for every model:
  model ID
  revision
  weight digest
  params
  input size
  feature dim
  feature extraction time
  peak memory
  5-NN metrics
  probe metrics
  best epoch
  artifact digest
  reload parity

overall wall time
cell success count
output inventory
```

---

# 75. Terminal summary

The final cell should print a compact table such as:

```text
DIMER Modern Image Classification Workshop
-------------------------------------------

Dataset:
  train: 108
  validation: 24
  test: 48
  classes: 6

Model             5NN Acc   Probe Acc   Macro F1   Feature Dim
----------------------------------------------------------------
ResNet-50          ...        ...         ...          ...
MobileNetV4        ...        ...         ...          ...
ConvNeXt-Tiny      ...        ...         ...          ...
ViT-B/16           ...        ...         ...          ...
SwinV2-Tiny        ...        ...         ...          ...
EVA-02 Base        ...        ...         ...          ...

Outputs:
  outputs/modern_image_classification/
```

Do not append:

```text
Winner: ...
Best model: ...
```

The tradeoff table should let the learner make that decision for their own deployment constraints.

---

# 76. Workshop learning arc

**Classic CNN**  
↓  
ResNet demonstrates deep residual convolution.

**Efficiency-first CNN**  
↓  
MobileNetV4 asks how much capability can fit into a small edge-oriented model.

**Modern ConvNet**  
↓  
ConvNeXt shows how convolutional networks evolved using transformer-era design ideas.

**Global transformer**  
↓  
ViT treats the image as a token sequence.

**Hierarchical transformer**  
↓  
SwinV2 restores locality and multiscale hierarchy through shifted windows.

**High-capacity modern transformer**  
↓  
EVA-02 combines higher resolution, masked-image pretraining and a modern transformer recipe.

**Common frozen representation**  
↓  
All six become feature extractors.

**Same linear probe**  
↓  
The downstream classification protocol is controlled.

**Data efficiency + resource cost**  
↓  
Quality is interpreted alongside model size, latency and labelled-data requirements.

**Deployment lesson**  
↓  
There is no universally optimal classifier: checkpoint choice depends on the representation quality, compute envelope, data availability and target domain.