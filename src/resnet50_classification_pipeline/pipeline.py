"""ImageNet-1k classification with the pinned ``timm/resnet50.a1_in1k`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. Preprocessing is
the upstream ``pretrained_cfg`` (resize/crop/normalize) resolved through ``timm.data``.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

MODEL_ID = "timm/resnet50.a1_in1k"
MODEL_REVISION = "767268603ca0cb0bfe326fa87277f19c419566ef"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "resnet50-a1"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

NUM_CLASSES = 1000
MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work
MAX_BATCH = 64  # images per predict() call
DEFAULT_TOP_K = 5
DECISION_RULE = "argmax"  # the label reported as `predicted_index` is the softmax argmax; no threshold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _hub_reference(model_id: str, revision: str) -> str:
    """timm's ``hf-hub:owner/name@revision`` form; ``hf_split`` passes ``revision=`` to hf_hub_download."""
    return f"hf-hub:{model_id}@{revision}"


def top_k_accuracy(predictions: Sequence[Any], targets: Sequence[int], k: int = 1) -> float:
    """Fraction of items whose target index is among the first ``k`` predicted indices.

    ``predictions`` may be the per-image dicts returned by ``predict`` or plain index sequences.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hits = 0
    for pred, target in zip(predictions, targets, strict=True):
        ranked = pred["top_k"] if isinstance(pred, Mapping) else pred
        indices = [int(item["index"]) if isinstance(item, Mapping) else int(item) for item in ranked]
        hits += int(target in indices[:k])
    return hits / len(predictions)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image or a sequence of them; any mode, converted to RGB",
    "image_side_px": [1, MAX_IMAGE_SIDE],
    "batch": [1, MAX_BATCH],
    "top_k": [1, NUM_CLASSES],
    "preprocessing": "resize to 235 px, center-crop 224x224 (crop_pct 0.95, bicubic), ImageNet normalisation",
}


def _check_inputs(images: Any, top_k: int, max_classes: int = NUM_CLASSES) -> list[Image.Image]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the images as a list."""
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be between 1 and MAX_BATCH={MAX_BATCH}, got {len(images)}")
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= max_classes:
        raise ValueError(f"top_k must be between 1 and {max_classes}")
    return list(images)


def validate_inputs(
    images: Image.Image | Sequence[Image.Image],
    top_k: int = DEFAULT_TOP_K,
    *,
    names: Sequence[str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``predict`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    checked = _check_inputs(images, top_k, max_classes=num_classes)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)}
            for i, image in enumerate(checked)
        ],
        "top_k": top_k,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], targets: Sequence[int] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``targets`` (one ImageNet-1k index per prediction) the report carries ``top_k_accuracy``
    at k=1 and k=5 as sample-sanity evidence; without them the verdict is ``not-measurable`` and
    the report says what labelled data would make the task measurable.
    """
    predictions = result["predictions"]
    base = {
        "task": "imagenet-1k single-label classification",
        "decision_rule": result.get("decision_rule", DECISION_RULE),
        "sample_kind": sample_kind,
        "n_predictions": len(predictions),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if targets is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth class index was supplied for the evaluated images",
            "needs": (
                "labelled photographs with ImageNet-1k class indices (0-999), e.g. a held-out sample of your "
                "own data, scored with top_k_accuracy against the majority-class baseline of that sample"
            ),
        }
    top_k = int(result.get("top_k", DEFAULT_TOP_K))
    ks = sorted({1, min(5, top_k)})
    return {
        **base,
        "metrics": [
            {
                "id": "top_k_accuracy",
                "k": k,
                "value": top_k_accuracy(predictions, list(targets), k=k),
                "estimation": "single sample, no dispersion estimate",
            }
            for k in ks
        ],
        "verdict": "sample-sanity",
        "reason": f"{len(predictions)} labelled image(s) from the tutorial sample; not a benchmark",
        "needs": "a labelled evaluation set from the deployment domain for any generalisable accuracy claim",
    }


# --- Fine-tuning helpers: trainable set, held-out evaluation with its uncertainty, dataset intake ---

MAX_DATASET_ARCHIVE_BYTES = 200_000_000  # a BYOD .zip larger than this is refused before it is opened
MAX_DATASET_IMAGES = 2000  # image files per dataset archive
MAX_DATASET_CLASSES = 100
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SPLIT_DIR_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val"}
# RN-M5: a user's test split is never pooled into training; an archive holding one is refused before decoding.
TEST_DIR_NAMES = {"test", "testing"}


def _check_training_config(epochs: Any, batch_size: Any, learning_rate: Any, weight_decay: Any) -> None:
    """Raise ValueError naming the first training setting outside its accepted range."""
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float) or not learning_rate > 0:
        raise ValueError(f"learning_rate must be a positive number, got {learning_rate!r}")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, int | float) or weight_decay < 0:
        raise ValueError(f"weight_decay must be a number >= 0, got {weight_decay!r}")


def set_trainable(model: Any, *, train_backbone: bool) -> dict[str, Any]:
    """Apply the fine-tuning rule to a timm classifier and count what it trains.

    ``train_backbone=True`` leaves every parameter trainable (full fine-tuning); ``False`` freezes every
    parameter except those of ``model.get_classifier()`` (head-only fine-tuning).
    """
    if not isinstance(train_backbone, bool):
        raise TypeError("train_backbone must be True (full fine-tuning) or False (head only)")
    if not train_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "method": "full fine-tuning" if train_backbone else "head-only fine-tuning",
        "trainable": "all" if train_backbone else "head",
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def trainable_parameter_counts(num_classes: int, *, train_backbone: bool) -> dict[str, Any]:
    """The counts ``fit`` will train, computed on the bare architecture (no weights are read)."""
    import timm

    model = timm.create_model(MODEL_ID.split("/", 1)[1], pretrained=False, num_classes=num_classes)
    return set_trainable(model, train_backbone=train_backbone)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % with the default ``z``)."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n >= 1, got {successes}/{n}")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def finetune_evaluation_report(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    class_names: Sequence[str],
    *,
    item_ids: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    sample_kind: str = "tutorial",
) -> dict[str, Any]:
    """Held-out evaluation of a fine-tuned classifier, reported with its uncertainty.

    The verdict is ``sample-sanity`` (never a success claim). ``comparison_to_baseline`` is read from the 95 %
    Wilson interval of the accuracy against the majority-class baseline of the same items: ``above-baseline``
    only when the whole interval lies above the baseline, ``below-baseline`` when it lies below, otherwise
    ``indistinguishable-from-baseline``.
    """
    n = len(true_indices)
    if n < 1 or len(predicted_indices) != n:
        raise ValueError("true_indices and predicted_indices must be non-empty and of equal length")
    ids = list(item_ids) if item_ids is not None else [f"item-{i}" for i in range(n)]
    correct = sum(int(t == p) for t, p in zip(true_indices, predicted_indices, strict=True))
    accuracy = correct / n
    counts = {name: sum(1 for t in true_indices if t == i) for i, name in enumerate(class_names)}
    majority_class = max(counts, key=lambda name: (counts[name], -list(class_names).index(name)))
    baseline = counts[majority_class] / n
    low, high = wilson_interval(correct, n)
    if low > baseline:
        comparison = "above-baseline"
    elif high < baseline:
        comparison = "below-baseline"
    else:
        comparison = "indistinguishable-from-baseline"
    per_class = {
        name: {
            "total": counts[name],
            "correct": sum(
                1 for t, p in zip(true_indices, predicted_indices, strict=True) if t == i and p == i
            ),
        }
        for i, name in enumerate(class_names)
    }
    misclassified = [
        {
            "id": ids[k],
            "true_label": class_names[t],
            "predicted_label": class_names[p],
            **({"score": float(scores[k])} if scores is not None else {}),
        }
        for k, (t, p) in enumerate(zip(true_indices, predicted_indices, strict=True))
        if t != p
    ]
    return {
        "task": "single-label classification fine-tuning evaluation",
        "verdict": "sample-sanity",
        "sample_kind": sample_kind,
        "n": n,
        "correct": correct,
        "classes": list(class_names),
        "metrics": {
            "accuracy": accuracy,
            "accuracy_wilson_95": [low, high],
            "majority_baseline_accuracy": baseline,
            "majority_baseline_class": majority_class,
            "accuracy_minus_baseline": accuracy - baseline,
        },
        "comparison_to_baseline": comparison,
        "per_class": per_class,
        "misclassified": misclassified,
        "estimation": (
            f"tutorial metric on {n} held-out image(s), not a benchmark: one seeded split, one training run, "
            "95 % Wilson score interval for the accuracy; one image moves the accuracy by "
            f"{100 / n:.1f} percentage points"
        ),
        "needs": "a larger labelled held-out set from the deployment domain for any accuracy claim",
    }


def synthetic_stripes_dataset(side: int = 64, per_class: int = 6, n_val: int = 2) -> dict[str, Any]:
    """Deterministic two-class fallback (horizontal vs vertical stripes) for a runtime without the dataset."""
    import numpy as np

    if not 1 <= n_val < per_class:
        raise ValueError("need 1 <= n_val < per_class")
    classes = ["synthetic_horizontal_stripe", "synthetic_vertical_stripe"]
    keys = ("train_images", "train_targets", "val_images", "val_targets")
    split: dict[str, list[Any]] = {key: [] for key in keys}
    ids: dict[str, list[str]] = {"train": [], "val": []}
    digest = hashlib.sha256()
    for cls_idx, pattern in enumerate(("horizontal", "vertical")):
        for i in range(per_class):
            arr = np.zeros((side, side, 3), dtype=np.uint8)
            if pattern == "horizontal":
                arr[::16, :, 0] = 255
            else:
                arr[:, ::16, 1] = 255
            arr[:, :, 2] = (i * 30) % 255
            digest.update(arr.tobytes())
            part = "train" if i < per_class - n_val else "val"
            split[f"{part}_images"].append(Image.fromarray(arr))
            split[f"{part}_targets"].append(cls_idx)
            ids[part].append(f"{pattern}_{i}")
    return {
        **split,
        "classes": classes,
        "train_ids": ids["train"],
        "val_ids": ids["val"],
        "manifest": {
            "source": "synthetic stripes fallback (generated in code)",
            "sha256": digest.hexdigest(),
            "layout": "generated",
            "classes": classes,
            "per_class": {c: {"train": per_class - n_val, "val": n_val} for c in classes},
            "skipped": [],
        },
    }


def _decode_member(archive: Any, member: str) -> Image.Image:
    """Open one archive member, check its size against MAX_IMAGE_SIDE before decoding, then decode to RGB."""
    try:
        image = Image.open(io.BytesIO(archive.read(member)))
        validate_inputs(image, top_k=1, names=[member], num_classes=1)
        return image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(
            f"{member}: not a decodable image ({type(exc).__name__}); PNG/JPEG/WebP/BMP expected. "
            "Remove or replace it and upload again"
        ) from None
    except ValueError as exc:
        raise ValueError(f"{member}: {exc}; resize or remove it and upload again") from None


def load_image_zip(
    zip_bytes: bytes,
    *,
    seed: int,
    validation_split: float = 0.2,
    subset_per_class: int | None = None,
    source: str = "uploaded archive",
) -> dict[str, Any]:
    """Read a class-folder image archive into a seeded train/val split, refusing bad layouts before training.

    Two layouts are accepted: ``<class>/<image>`` folders, split here, or ``train/<class>/...`` plus
    ``val/<class>/...`` (``valid``/``validation`` also accepted) used as given.

    The split here is **pair-grouped** and seeded, per class. Images of one class whose file names
    (the last path component) are equal form one group, e.g. ``original_images/frog/image_7.png`` and
    ``darkened_images/frog/image_7.png``; a group always lands on one side of the split. Groups are
    listed in archive order of their first image and shuffled with one ``random.Random(seed)`` shared
    across classes (classes in archive order). With ``subset_per_class`` the leading groups are kept
    while their image total stays within ``subset_per_class``. The first
    ``max(1, int(n_groups * validation_split))`` kept groups are validation, the rest training. When no
    two images share a name this equals the previous per-image split. Every image must sit
    inside a class folder, every validation class must exist in ``train`` and every train class in
    ``val``, a ``test/`` folder is refused (never pooled into training), at least 2 classes are needed,
    every image is decoded and checked against ``MAX_IMAGE_SIDE`` here, and the archive limits are
    ``MAX_DATASET_ARCHIVE_BYTES``, ``MAX_DATASET_IMAGES`` and ``MAX_DATASET_CLASSES``. Each refusal names the
    file or class and the rule.
    """
    if len(zip_bytes) > MAX_DATASET_ARCHIVE_BYTES:
        raise ValueError(
            f"archive is {len(zip_bytes)} bytes; limit MAX_DATASET_ARCHIVE_BYTES={MAX_DATASET_ARCHIVE_BYTES}"
        )
    if not 0 < validation_split < 1:
        raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise ValueError(f"{source}: not a readable .zip archive") from None
    with archive:
        skipped: list[dict[str, str]] = []
        names: list[str] = []
        for info in archive.infolist():
            name = info.filename
            unsafe = ".." in name.replace("\\", "/").split("/") or name.startswith(("/", "\\"))
            if unsafe or ":" in name.split("/")[0]:
                raise ValueError(f"{name}: illegal path (absolute or '..'); refusing the archive")
            if info.is_dir():
                continue
            if name.startswith("__MACOSX") or name.rsplit("/", 1)[-1].startswith("."):
                skipped.append({"file": name, "reason": "system metadata"})
            elif name.lower().endswith(IMAGE_SUFFIXES):
                names.append(name)
            else:
                skipped.append({"file": name, "reason": f"not an image ({', '.join(IMAGE_SUFFIXES)})"})
        if not names:
            raise ValueError(f"{source}: no image files ({', '.join(IMAGE_SUFFIXES)}) found")
        if len(names) > MAX_DATASET_IMAGES:
            raise ValueError(f"{source}: {len(names)} images; limit MAX_DATASET_IMAGES={MAX_DATASET_IMAGES}")

        def parts_of(name: str) -> list[str]:
            return name.strip("/").split("/")

        for name in names:
            if any(p.lower() in TEST_DIR_NAMES for p in parts_of(name)[:-1]):
                raise ValueError(
                    f"{name}: the archive has a test/ folder. This notebook holds out its own validation "
                    "split and would otherwise pool your test images into training; remove test/ (keep it "
                    "for a final check outside this notebook) or rename it to val/"
                )

        def split_of(name: str) -> str | None:
            found = {SPLIT_DIR_NAMES[p.lower()] for p in parts_of(name)[:-1] if p.lower() in SPLIT_DIR_NAMES}
            return found.pop() if len(found) == 1 else ("ambiguous" if found else None)

        def class_of(name: str) -> str:
            parts = parts_of(name)
            if len(parts) < 2 or parts[-2].lower() in SPLIT_DIR_NAMES:
                raise ValueError(
                    f"{name}: image is not inside a class folder; put every image in <class>/ "
                    "(or train/<class>/ and val/<class>/)"
                )
            return parts[-2]

        splits = {name: split_of(name) for name in names}
        if any(s == "ambiguous" for s in splits.values()):
            bad = next(n for n, s in splits.items() if s == "ambiguous")
            raise ValueError(f"{bad}: path names both a train and a val folder")
        presplit = "train" in splits.values() and "val" in splits.values()
        if presplit and any(s is None for s in splits.values()):
            bad = next(n for n, s in splits.items() if s is None)
            raise ValueError(f"{bad}: the archive has train/ and val/ folders, so every image must be in one")

        files: dict[str, dict[str, list[str]]] = {}
        for name in names:
            part = splits[name] if presplit else "all"
            files.setdefault(class_of(name), {}).setdefault(part, []).append(name)
        if presplit:
            classes = sorted(c for c, parts in files.items() if parts.get("train"))
            missing = sorted(c for c, parts in files.items() if not parts.get("train"))
            if missing:
                raise ValueError(
                    f"class(es) {missing} appear in val/ but not in train/; each val class needs train images"
                )
        else:
            classes = sorted(files)
        if len(classes) < 2:
            raise ValueError(f"classification needs at least 2 classes, found {len(classes)}: {classes}")
        if len(classes) > MAX_DATASET_CLASSES:
            raise ValueError(f"{len(classes)} classes; limit MAX_DATASET_CLASSES={MAX_DATASET_CLASSES}")
        index = {c: i for i, c in enumerate(classes)}

        chosen: dict[str, dict[str, list[str]]] = {}
        if presplit:
            if not any(parts.get("val") for parts in files.values()):
                raise ValueError("val/ holds no images")
            no_val = sorted(c for c in classes if not files[c].get("val"))
            if no_val:
                raise ValueError(
                    f"class(es) {no_val} appear in train/ but not in val/; every class needs >= 1 val image "
                    "so its accuracy can be measured"
                )
            chosen = {c: {"train": files[c]["train"], "val": files[c].get("val", [])} for c in files}
        else:
            rng = random.Random(seed)
            for cls, parts in files.items():
                grouped: dict[str, list[str]] = {}
                for name in parts["all"]:
                    grouped.setdefault(parts_of(name)[-1], []).append(name)
                groups = list(grouped.values())
                if len(groups) < 2:
                    raise ValueError(
                        f"class {cls!r} has {len(groups)} image group(s) (images sharing a file name are "
                        "one group); each class needs >= 2 for a train/val split"
                    )
                rng.shuffle(groups)
                if subset_per_class:
                    kept, total = [], 0
                    for group in groups:
                        if total + len(group) > subset_per_class:
                            break
                        kept.append(group)
                        total += len(group)
                    if len(kept) < 2:
                        raise ValueError(f"class {cls!r}: subset_per_class={subset_per_class} < 2 groups")
                    groups = kept
                n_val = max(1, int(len(groups) * validation_split))
                chosen[cls] = {
                    "train": [name for group in groups[n_val:] for name in group],
                    "val": [name for group in groups[:n_val] for name in group],
                }

        out: dict[str, Any] = {"train_images": [], "train_targets": [], "val_images": [], "val_targets": [],
                               "train_ids": [], "val_ids": []}
        for cls, parts in chosen.items():
            for part in ("train", "val"):
                for member in parts[part]:
                    out[f"{part}_images"].append(_decode_member(archive, member))
                    out[f"{part}_targets"].append(index[cls])
                    out[f"{part}_ids"].append(member)
    out["classes"] = classes
    out["manifest"] = {
        "source": source,
        "sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "bytes": len(zip_bytes),
        "layout": "train/ and val/ folders" if presplit else "class folders, seeded pair-grouped split",
        "validation_split": None if presplit else validation_split,
        "seed": None if presplit else seed,
        "subset_per_class": None if presplit else subset_per_class,
        "classes": classes,
        "per_class": {c: {"train": len(chosen[c]["train"]), "val": len(chosen[c]["val"])} for c in classes},
        "images_in_archive": len(names),
        "skipped": skipped,
    }
    return out


@dataclass
class ResNet50ClassificationPipeline:
    """``_runner`` maps a float tensor (N, 3, H, W) to logits (N, NUM_CLASSES); injectable for tests."""

    _runner: Callable[[Any], Any]
    _transform: Callable[[Image.Image], Any]
    device: str = "cpu"
    labels: tuple[str, ...] = ()
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ResNet50ClassificationPipeline:
        import timm
        import torch
        from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        arch_name = MODEL_ID.split("/", 1)[1]
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        if (root / "model-config.json").is_file() and (root / WEIGHTS_FILE).is_file():
            from safetensors.torch import load_file
            with open(root / "model-config.json", encoding="utf-8") as fh:
                cfg = json.load(fh)
            num_classes = cfg.get("num_classes", len(cfg.get("class_names", [])))
            labels = tuple(cfg.get("class_names", [f"class_{i}" for i in range(num_classes)]))
            model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
            model.load_state_dict(load_file(root / WEIGHTS_FILE, device=str(resolved_device)), strict=True)
            source = "fine-tuned-artifact"
            data_config = cfg.get("data_config") or resolve_model_data_config(model)
            transform = create_transform(**data_config, is_training=False)
        elif (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / CONFIG_FILE, encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot_name = f"{config['architecture']}.{config['pretrained_cfg']['tag']}"
            if snapshot_name != arch_name:
                raise ValueError(f"snapshot config names {snapshot_name!r}, expected {arch_name!r}")
            overlay = dict(config["pretrained_cfg"])
            overlay["file"] = str(root / WEIGHTS_FILE)  # 'file' takes precedence over hf_hub_id in timm
            model = timm.create_model(
                arch_name, pretrained=True, pretrained_cfg_overlay=overlay, num_classes=NUM_CLASSES
            )
            source = "local-snapshot"
            data_config = resolve_model_data_config(model)
            transform = create_transform(**data_config, is_training=False)
            info = ImageNetInfo(subset="imagenet-1k")
            labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION), pretrained=True, num_classes=NUM_CLASSES
            )
            source = "hf-hub"
            data_config = resolve_model_data_config(model)
            transform = create_transform(**data_config, is_training=False)
            info = ImageNetInfo(subset="imagenet-1k")
            labels = tuple(info.index_to_description(i) for i in range(info.num_classes()))
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        model = model.eval().to(resolved_device)

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        return cls(runner, transform, resolved_device, labels, source)

    @classmethod
    def fit(
        cls,
        train_images: Sequence[Image.Image],
        train_targets: Sequence[int],
        val_images: Sequence[Image.Image],
        val_targets: Sequence[int],
        class_names: Sequence[str],
        *,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        seed: int = 20260910,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        allow_download: bool = False,
        train_backbone: bool = True,
        provenance: Mapping[str, Any] | None = None,
    ) -> tuple[ResNet50ClassificationPipeline, dict[str, Any]]:
        """Fine-tune on custom classes in-process: a new linear classifier sized to ``class_names``.

        ``train_backbone=True`` (the default) is **full fine-tuning**: AdamW updates every parameter, the
        pretrained backbone included. ``train_backbone=False`` freezes everything except the new classifier
        (``model.get_classifier()``) and keeps every BatchNorm layer in eval mode, so its running statistics
        stay pretrained: head-only fine-tuning, where only ``fc.*`` changes. The training configuration, the
        trainable/frozen parameter counts and ``provenance`` (e.g. the dataset source, digest and split sizes)
        are returned under ``config`` and written into ``model-config.json`` with the artifact.
        """
        import timm
        import torch
        import torch.nn.functional as F
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader, Dataset

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = len(class_names)
        if num_classes < 2:
            raise ValueError(f"classification requires at least 2 classes, got {num_classes}")
        _check_training_config(epochs, batch_size, learning_rate, weight_decay)
        arch_name = MODEL_ID.split("/", 1)[1]

        # timm's training transforms (random-resized crop, flip, interpolation) use from Python's `random`, so
        # it is seeded too: with only torch seeded, two CPU runs with the same seed trained differently.
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / CONFIG_FILE, encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot_name = f"{config['architecture']}.{config['pretrained_cfg']['tag']}"
            if snapshot_name != arch_name:
                raise ValueError(f"snapshot config names {snapshot_name!r}, expected {arch_name!r}")
            overlay = dict(config["pretrained_cfg"])
            overlay["file"] = str(root / WEIGHTS_FILE)
            model = timm.create_model(
                arch_name,
                pretrained=True,
                pretrained_cfg_overlay=overlay,
                num_classes=num_classes,
            )
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION),
                pretrained=True,
                num_classes=num_classes,
            )
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )

        model.to(resolved_device)
        data_config = timm.data.resolve_model_data_config(model)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        eval_transform = timm.data.create_transform(**data_config, is_training=False)

        class ImageDataset(Dataset):
            def __init__(self, imgs: Sequence[Image.Image], targets: Sequence[int], transform_fn: Any):
                self.imgs = list(imgs)
                self.targets = list(targets)
                self.transform_fn = transform_fn

            def __len__(self) -> int:
                return len(self.imgs)

            def __getitem__(self, idx: int) -> tuple[Any, int]:
                img = self.imgs[idx].convert("RGB")
                tensor = self.transform_fn(img)
                return tensor, self.targets[idx]

        train_loader = DataLoader(
            ImageDataset(train_images, train_targets, train_transform),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ImageDataset(val_images, val_targets, eval_transform),
            batch_size=batch_size,
            shuffle=False,
        )

        counts = set_trainable(model, train_backbone=train_backbone)
        config = {
            "method": counts["method"],
            "trainable": counts["trainable"],
            "trainable_parameters": counts["trainable_parameters"],
            "frozen_parameters": counts["frozen_parameters"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "batchnorm_statistics": "updated in training" if train_backbone else "frozen (eval mode)",
            "optimizer": "AdamW",
            "loss": "cross-entropy",
            "train_samples": len(train_targets),
            "val_samples": len(val_targets),
            **({"dataset": dict(provenance)} if provenance else {}),
        }
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        history = []

        for epoch in range(epochs):
            model.train()
            if not train_backbone:
                # RN-M3: a frozen backbone stays frozen. ResNet-50's BatchNorm running statistics update in
                # train() mode even with requires_grad False, so those layers are kept in eval() mode.
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            train_loss_sum, train_count = 0.0, 0
            for inputs, targets in train_loader:
                inputs = inputs.to(resolved_device)
                targets = targets.to(resolved_device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach().cpu()) * targets.numel()
                train_count += targets.numel()

            model.eval()
            val_loss_sum, val_correct, val_count = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(resolved_device)
                    targets = targets.to(resolved_device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    val_loss_sum += float(loss.detach().cpu()) * targets.numel()
                    val_correct += int((outputs.argmax(dim=-1) == targets).sum())
                    val_count += targets.numel()

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss_sum / max(1, train_count),
                "val_loss": val_loss_sum / max(1, val_count),
                "val_accuracy": val_correct / max(1, val_count),
            })

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            safetensors_path = out_path / WEIGHTS_FILE
            tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(tensors, safetensors_path)
            config_payload = {
                "architecture": "resnet50",
                "num_classes": num_classes,
                "class_names": list(class_names),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "data_config": data_config,
                "fine_tuning": config,
            }
            with open(out_path / "model-config.json", "w", encoding="utf-8") as fh:
                json.dump(config_payload, fh, indent=2)

        model.eval()

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        pipeline = cls(runner, eval_transform, resolved_device, tuple(class_names), "fine-tuned")
        return pipeline, {
            "history": history, "class_names": list(class_names), "device": resolved_device, "config": config
        }

    def _validate(self, images: Any, top_k: int) -> list[Image.Image]:
        max_classes = len(self.labels) if self.labels else NUM_CLASSES
        return _check_inputs(images, top_k, max_classes=max_classes)

    def predict(
        self, images: Image.Image | Sequence[Image.Image], top_k: int | None = None
    ) -> dict[str, Any]:
        """Classify images; ``score`` is a softmax score, not a calibrated probability."""
        import torch

        active_classes = len(self.labels) if self.labels else NUM_CLASSES
        resolved_top_k = min(DEFAULT_TOP_K, active_classes) if top_k is None else top_k
        batch_images = self._validate(images, resolved_top_k)
        batch = torch.stack([self._transform(image.convert("RGB")) for image in batch_images])
        logits = self._runner(batch)
        expected_classes = active_classes
        if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch_images), expected_classes):
            raise RuntimeError(f"runner must return a tensor of shape (batch, {expected_classes})")
        scores = torch.softmax(logits.float(), dim=-1).cpu()
        values, indices = torch.topk(scores, k=resolved_top_k, dim=-1)
        predictions = []
        for image_values, image_indices in zip(values.tolist(), indices.tolist(), strict=True):
            image_values = [float(s) for s in image_values]
            ranked = [
                {"label": self.labels[i] if i < len(self.labels) else str(i), "index": i, "score": s}
                for s, i in zip(image_values, image_indices, strict=True)
            ]
            best = ranked[0]
            predictions.append(
                {"predicted_index": best["index"], "predicted_label": best["label"], "top_k": ranked}
            )
        return {
            "predictions": predictions,
            "top_k": resolved_top_k,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
