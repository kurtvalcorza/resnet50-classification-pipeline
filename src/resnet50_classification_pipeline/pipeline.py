"""ImageNet-1k classification with the pinned ``timm/resnet50.a1_in1k`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or,
when explicitly allowed, from the Hugging Face Hub at the pinned revision. Preprocessing is
the upstream ``pretrained_cfg`` (resize/crop/normalize) resolved through ``timm.data``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

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
    ) -> tuple[ResNet50ClassificationPipeline, dict[str, Any]]:
        """Fine-tune the ResNet-50 model on custom classes 100% in-kernel."""
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
        arch_name = MODEL_ID.split("/", 1)[1]

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        history = []

        for epoch in range(epochs):
            model.train()
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
            }
            with open(out_path / "model-config.json", "w", encoding="utf-8") as fh:
                json.dump(config_payload, fh, indent=2)

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))

        pipeline = cls(runner, eval_transform, resolved_device, tuple(class_names), "fine-tuned")
        return pipeline, {"history": history, "class_names": list(class_names), "device": resolved_device}

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
