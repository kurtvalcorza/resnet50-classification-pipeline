# ResNet-50 Classification Pipeline

DIMER inference wrapper for **`timm/resnet50.a1_in1k`** — ImageNet-1k image classification (1000 classes) — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `timm/resnet50.a1_in1k`
- Revision: `767268603ca0cb0bfe326fa87277f19c419566ef`
- Upstream weight license: Apache-2.0
- Upstream task: image classification, 1000 ImageNet-1k classes, 224×224 eval input
- Repository adaptation: **none**; inference only

## Quick start

```python
from PIL import Image
from resnet50_classification_pipeline import ResNet50ClassificationPipeline, top_k_accuracy

pipe = ResNet50ClassificationPipeline.from_pretrained()          # cuda:0 if available, else cpu
result = pipe.predict(Image.open("photo.jpg"), top_k=5)
print(result["predictions"][0]["predicted_label"], result["predictions"][0]["top_k"][0]["score"])
print(top_k_accuracy(result["predictions"], [targets_index], k=1))
```

`score` is a softmax score over 1000 classes, not a calibrated probability; the reported label is the argmax.

## Weights layout

```
weights/resnet50-a1/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # timm pretrained_cfg: input size, mean/std, crop
  model.safetensors          # 102469840 bytes, git-ignored
```

`from_pretrained()` calls `verify_snapshot()` first and refuses to load if any file is missing or its SHA-256 differs from the manifest. Without a snapshot, `allow_download=True` loads from the Hub through timm's `hf-hub:timm/resnet50.a1_in1k@767268603ca0cb0bfe326fa87277f19c419566ef` form; the default is to refuse. To stage the snapshot: `hf download timm/resnet50.a1_in1k --revision 767268603ca0cb0bfe326fa87277f19c419566ef --local-dir weights/resnet50-a1`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed
python smoke.py                  # loads the snapshot, classifies one synthetic image
```

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.0 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md`.
