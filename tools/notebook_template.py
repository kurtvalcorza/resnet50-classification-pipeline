"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "resnet50_classification_pipeline",
    "repo_name": "resnet50-classification-pipeline",
    "stem": "resnet50_classification",
    "notebook_name": "resnet50_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the pinned "
        "snapshot, generates the deterministic synthetic sample image, validates it into an input manifest, classifies it with the "
        "pretrained head, writes the zero-shot evaluation report, downloads and digest-verifies the `Cleanlab/cifar-10-subset` tutorial "
        "dataset (a balanced per-class subset with a seeded 80/20 split), performs a bounded in-kernel fine-tuning run of a new "
        "classification head with AdamW and cross-entropy, exports the artifact (`model.safetensors` + `model-config.json`), reloads it "
        "from disk across a fresh boundary and evaluates the reloaded model on the held-out split against a majority-class baseline, "
        "and writes outputs and provenance. No repository clone, DIMER worker or service, credential, upload dialog or configuration "
        "edit is required (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "Two optional branches, both off by default and never part of the default path: `USE_BYOD = True` in Section 4 reads one "
        "image (from `BYOD_IMAGE_PATH`, or through the Colab upload dialog when the path is empty) that passes through the same validation, classification, evaluation-report and export cells as the synthetic sample; "
        "`USE_BYOD_DATASET = True` in Section 8 reads a `.zip` of class folders (from `BYOD_DATASET_PATH`, or the upload dialog) (or `train/` and `val/` directories) that enters the "
        "same validation, seeded split, in-kernel fine-tuning, export, fresh-reload and held-out evaluation cells as the tutorial "
        "dataset. Expected formats, limits and privacy guidance are stated in the Prerequisites and in those cells; uploads stay "
        "inside this runtime."
    ),
    "pipeline_class": "ResNet50ClassificationPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python and PIL, has used Colab or Jupyter, and wants to see what an '
            "ImageNet classifier's output does and does not mean, how a prediction is validated and reported honestly, and how the same backbone is adapted to new "
            'classes with a new head. No prior experience with timm or fine-tuning is assumed; terms are explained where they first matter and again in the '
            '**Glossary** at the end. CPU is adequate; a GPU runtime is faster.\n\n**Input → Model → Output.**\n\n| | Classification | Head fine-tuning |\n|---|---|---|\n| '
            'Input | one RGB image (any size up to 4096 px; resized and center-cropped to 224 × 224) | labelled images in class folders: the pinned 2-class '
            'CIFAR-10 subset (`frog`, `truck`), or your `.zip` |\n| Model | timm ResNet-50 `a1_in1k`, 1000-class head | the same backbone with a new linear head '
            'sized to your classes, trained with AdamW and cross-entropy |\n| Output | argmax label and a top-5 list of uncalibrated softmax scores; an evaluation '
            'report (`not-measurable` without a label) | `model.safetensors` + `model-config.json`, reloaded from disk and scored on the held-out split beside the '
            'majority-class baseline |\n\n**How to use this notebook.** Choose a runtime (CPU works; **Runtime → Change runtime type → T4 GPU** is faster), then '
            "**Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 "
            'are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them '
            'without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited. Before each principal '
            'result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning**. The recorded Kaggle run in '
            '`docs/release-verification.md` records a pass but no per-stage numbers, so the worked answers describe what to look for rather than quoting values. '
            '**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 '
            'infrastructure → 4 the synthetic sample or your image → 5 input validation *(core concept: the input contract)* → 6 classification *(core concept: '
            'scores are not probabilities)* → 7 the evaluation report *(evaluation practice)* → 8 dataset and head fine-tuning → 9 fresh reload and held-out '
            'evaluation *(evaluation practice)* → 10 outputs and provenance *(engineering)* → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "weights_key": "resnet50-a1",
    "runtime_imports": ["torch", "timm"],
    "title": "ResNet-50 a1_in1k — DIMER image classification tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/resnet50-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/resnet50-classification-pipeline/blob/main/tutorials/resnet50_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fresnet50.a1__in1k-ffcc4d?style=flat",
            "https://huggingface.co/timm/resnet50.a1_in1k",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Fpytorch--image--models-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/pytorch-image-models",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2110.00476-b31b1b.svg", "https://arxiv.org/abs/2110.00476"),
    ],
    "capability": "ImageNet-1k single-label image classification (1000 classes) and in-kernel fine-tuning using the pinned `timm/resnet50.a1_in1k` weights",
    "intro": (
        "At inference the network maps one normalized 3×224×224 tensor to 1000 logits in a single forward pass; the "
        "pipeline applies a softmax and reports the argmax class plus the top-k classes with their scores. "
        "**In-kernel fine-tuning:** this notebook demonstrates both zero-shot base inference on ImageNet-1k classes and "
        "100% in-kernel classification head fine-tuning on custom classes using PyTorch AdamW optimization and cross-entropy "
        "loss. The upstream checkpoint supplies the pretrained backbone weights, and the carried pipeline module adds snapshot "
        "verification, input validation, head adaptation via `fit`, fresh-boundary reload verification, and the "
        "`top_k_accuracy`, `validate_inputs` and `evaluation_report` helpers. The default sample is a synthetic image "
        "generated in code; its prediction is demonstration (plumbing) evidence, not a production-quality or benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, generate a synthetic default input and validate it into an input manifest, "
        "run the supported task, read the argmax decision and the uncalibrated top-k softmax scores correctly, execute "
        "100% in-kernel fine-tuning on custom classes with AdamW and cross-entropy, export and fresh-reload fine-tuned "
        "artifacts, exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` only when a "
        "ground-truth class index exists and `not-measurable` otherwise, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, OCR, open-vocabulary classification, feature/embedding "
        "extraction. The base label space is fixed to the 1000 ImageNet-1k classes; an image whose subject is outside that "
        "space still receives a label unless adapted via in-kernel fine-tuning."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. Building the isolated environment (the pinned `torch==2.14.0` is its largest package; reused on a re-run) is the slowest setup step.",
        "- **Knowledge:** basic Python and PIL image handling; what a softmax over class logits is.",
        "- **Data:** the default sample is a deterministic 256×256 RGB gradient generated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, longest side at most 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 256×256 RGB gradient built in code (red ramps left to "
                "right, green top to bottom, blue is their mean), so it needs no download and its SHA-256 is printed for the "
                "record. A gradient is not a photograph of any ImageNet class, so it has **no ground truth**: whatever label "
                "the model returns is a sanity check that the input contract, preprocessing and forward pass work, not a "
                "correctness measurement. BYOD is optional and disabled by default; when enabled, upload one image file and, "
                "if you know its ImageNet-1k class index (0–999), set `GROUND_TRUTH_INDEX` so the evaluation step can compute "
                "`top_k_accuracy`. Leave it at `-1` when the label is unknown. Look for a dictionary naming the sample kind, "
                "its size and digest, and whether a ground-truth index was supplied.\n\n"
                "**Predict before running:** the default image is a colour gradient. Which of the 1000 ImageNet classes is it?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "GROUND_TRUTH_INDEX = -1  # @param {{type:\"integer\"}}\n"
                "# An image file already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                "SAMPLE_SIDE = 256\n\n"
                "if USE_BYOD:\n"
                "    from pathlib import Path\n"
                "    if BYOD_IMAGE_PATH.strip():\n"
                "        image_path = Path(BYOD_IMAGE_PATH.strip()).expanduser()\n"
                "        if not image_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_IMAGE_PATH {{BYOD_IMAGE_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one image file Pillow can open.')\n"
                "        image_name, image_bytes = image_path.name, image_path.read_bytes()\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_IMAGE_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the image in the runtime and set BYOD_IMAGE_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one image file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        image_name, image_bytes = next(iter(uploaded.items()))\n"
                "    try:\n"
                "        image = Image.open(io.BytesIO(image_bytes))\n"
                "        image.load()\n"
                "    except Exception as exc:\n"
                "        raise ValueError(f'{{image_name}}: Pillow cannot decode this file as an image ({{exc}}).') from None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic gradient: no randomness, so no seed is needed and the digest is stable.\n"
                "    ramp = np.linspace(0.0, 255.0, SAMPLE_SIDE)\n"
                "    red = np.tile(ramp, (SAMPLE_SIDE, 1))\n"
                "    green = red.T\n"
                "    blue = (red + green) / 2.0\n"
                "    array = np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)\n"
                "    image = Image.fromarray(array, mode='RGB')\n"
                "    image_name = f'synthetic_gradient_{{SAMPLE_SIDE}}.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "if GROUND_TRUTH_INDEX != -1 and not 0 <= GROUND_TRUTH_INDEX < NUM_CLASSES:\n"
                "    raise ValueError(f'GROUND_TRUTH_INDEX must be -1 (unknown) or an ImageNet-1k class index in 0..{{NUM_CLASSES - 1}}, got {{GROUND_TRUTH_INDEX}}.')\n"
                "ground_truth = None if GROUND_TRUTH_INDEX == -1 else GROUND_TRUTH_INDEX\n"
                "sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `sample_kind`, the image size and RGB digest, and `ground_truth_index` (None for the gradient).\n\n<details><summary>Check your '
                'reasoning</summary>None of them: a gradient is not a photograph of any class, so there is no right answer and no ground truth. Whatever label Section '
                '6 returns is plumbing evidence that the input contract and forward pass work, not a correctness measurement.</details>'
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `predict` "
                "applies — type, batch size 1..`MAX_BATCH`, image side 1..`MAX_IMAGE_SIDE` px, `top_k` 1..`NUM_CLASSES` — "
                "and returns an **input manifest** naming the schema and ceilings, each input's observed mode and size, "
                "and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a deliberately oversized image and records the pipeline's own error "
                "message as a finding. Inside the pipeline every accepted image is converted to RGB, resized to 235 px and "
                "center-cropped to 224×224 (`crop_pct = 0.95`, bicubic), so content near the border is cropped away; "
                "nothing else is dropped or altered.\n\n"
                "**Predict before running:** an image 4097 px wide — will `validate_inputs` resize it, or refuse it?"
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'NUM_CLASSES': NUM_CLASSES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}}}})\n"
                "input_manifest = validate_inputs(image, top_k=5, names=[image_name])\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice:** the ceilings, the accepted input's mode and size, and the `oversized-probe` finding with the pipeline's own message.\n\n<details><summary>Check "
                'your reasoning</summary>Refuse it: the side ceiling is `MAX_IMAGE_SIDE` (4096 px), and the rejection is recorded as a finding with the rule in the '
                'message, not swallowed. Accepted images are converted to RGB, resized and center-cropped to 224 × 224, so content near the border is lost.</details>'
            ),
        },
        {
            "md": (
                "## 6. Classify\n\n"
                "`predict` returns, per image, `predicted_index`/`predicted_label` and a `top_k` list of `{{label, index, "
                "score}}` entries **ordered by descending score** — rank position is the class ordering, and the exported "
                "files preserve it. The decision rule is `argmax` over the 1000 softmax scores (`decision_rule` in the "
                "result); the pipeline ships no acceptance threshold, and `score` is a softmax over uncalibrated logits, "
                "**not a calibrated probability**. A deployment that needs an abstain option must choose its own score "
                "cut-off on its own labelled data — downstream calibration is the caller's responsibility. Inference is "
                "deterministic given the same weights, device and library versions (no sampling, `model.eval()`); CPU, GPU "
                "and cuDNN kernel choices can reorder near-tied classes. Look for the ranked top-5 list; on the gradient "
                "expect a low top-1 score spread across unrelated classes.\n\n"
                "**Predict before running:** will the model refuse to answer for an image that is no ImageNet class?"
            ),
            "code": (
                "result = pipe.predict(image, top_k=5)\n"
                "prediction = result['predictions'][0]\n"
                "print({{'decision_rule': result['decision_rule'], 'predicted_index': prediction['predicted_index'], 'predicted_label': prediction['predicted_label'], 'device': result['device'], 'source': result['source']}})\n"
                "for rank, item in enumerate(prediction['top_k'], start=1):\n"
                "    print(f\"{{rank:>2}}. index {{item['index']:>4}}  score {{item['score']:.4f}}  {{item['label']}}\")"
            ),
        },
        {
            "md": (
                '**What to notice:** `decision_rule` (argmax), the top-5 list, and how spread out the scores are.\n\n<details><summary>Check your reasoning</summary>No. '
                'The model has no *none of these*: it always returns the argmax of a softmax over 1000 classes. On the gradient expect a low top-1 score spread across '
                'unrelated classes; a score is relative and uncalibrated, not the probability of being right.</details>'
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When a "
                "ground-truth class index was supplied in Section 4 it carries `top_k_accuracy` (the repository's metric "
                "helper) at k=1 and k=5 with the verdict `sample-sanity` — a single-image tutorial metric with no dispersion "
                "estimate. On the synthetic default sample no metric exists, so the verdict is `not-measurable` and the "
                "report states what would make the task measurable: labelled photographs with ImageNet-1k class indices, "
                "for example a held-out sample of your own data scored against its majority-class baseline, or the "
                "ImageNet-1k validation set (whose upstream 80.38 % top-1 / 94.60 % top-5 at 224 px is quoted from the "
                "upstream card, not measured here). The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "*Evaluation practice.* **Predict before running:** with no ground-truth label, what accuracy can the report give?"
            ),
            "code": (
                "targets = None if ground_truth is None else [ground_truth]\n"
                "report = evaluation_report(result, targets, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No ground-truth class index was supplied, so top_k_accuracy is not computed; the prediction above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                '**What to notice:** the `verdict` (`not-measurable` on the default sample) and the `needs` field naming what would make it measurable.\n\n<details><summary>Check '
                'your reasoning</summary>None: without a label there is nothing to compare, so the verdict is `not-measurable` rather than a made-up number. With a '
                'known class index through BYOD the report computes top-1 and top-5 for that one image, labelled `sample-sanity`.</details>'
            ),
        },
        {
            "md": (
                "## 8. Dataset acquisition and in-kernel fine-tuning\n\n"
                "`pipe.fit(...)` implements 100% in-kernel head adaptation: it replaces the 1000-class head with a new "
                "linear classifier sized to the target classes (`len(class_names)`), initializes from the verified backbone "
                "weights without shape collision, and trains with `torch.optim.AdamW` and cross-entropy loss. No external "
                "worker repositories, CLI subprocesses, or unpinned dependencies are invoked.\n\n"
                "**Data Acquisition & BYOD:**\n"
                "- **Default Sample Dataset (`USE_BYOD_DATASET = False`):** Automatically downloads and verifies the SHA-256 digest of "
                "[`Cleanlab/cifar-10-subset`](https://huggingface.co/datasets/Cleanlab/cifar-10-subset) (MIT license, ~986 KB, "
                "400 images across 2 balanced classes: `frog` and `truck`). A balanced subset is loaded for rapid tutorial smoke, "
                "and automatically split into `train` (80%) and `val` (20%). If the runtime is air-gapped, it gracefully falls back "
                "to deterministic synthetic stripes.\n"
                "- **Bring Your Own Data (`USE_BYOD_DATASET = True`):** Upload a `.zip` archive containing either explicit `train/` and `val/` "
                "directories or un-split class folders (in which case a seeded 80/20 stratified split is performed automatically). "
                "Enforces >= 2 classes and >= 2 images per class.\n\n"
                "Deployable fine-tuned artifacts (`model.safetensors` and `model-config.json`) are written atomically to `outputs/{stem}_finetuned`.\n\n"
                "**Predict before running:** the new head starts untrained. After one epoch on a few dozen images, will the fine-tuned model beat the majority-class baseline?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import random\n"
                "import urllib.request\n"
                "import zipfile\n\n"
                "USE_BYOD_DATASET = False  # @param {{type:\"boolean\"}}\n"
                "# A .zip already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_DATASET_PATH = ''  # @param {{type:\"string\"}}\n"
                "SAMPLE_DATASET_URL = 'https://huggingface.co/datasets/Cleanlab/cifar-10-subset/resolve/bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75/CIFAR-10-subset.zip'\n"
                "SAMPLE_DATASET_SHA256 = '66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19'\n"
                "SEED = 42\n"
                "VALIDATION_SPLIT = 0.2\n"
                "SUBSET_PER_CLASS = 16  # balanced sample per class for fast in-kernel smoke\n\n"
                "train_images, train_targets = [], []\n"
                "val_images, val_targets = [], []\n"
                "zip_bytes = None\n\n"
                "if USE_BYOD_DATASET:\n"
                "    from pathlib import Path\n"
                "    if BYOD_DATASET_PATH.strip():\n"
                "        zip_path = Path(BYOD_DATASET_PATH.strip()).expanduser()\n"
                "        if not zip_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_DATASET_PATH {{BYOD_DATASET_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one .zip of class folders.')\n"
                "        zip_name, zip_bytes = zip_path.name, zip_path.read_bytes()\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD_DATASET is True but BYOD_DATASET_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the .zip in the runtime and set BYOD_DATASET_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one dataset .zip archive (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        zip_name, zip_bytes = next(iter(uploaded.items()))\n"
                "    if not zip_name.lower().endswith('.zip'):\n"
                "        raise ValueError(f'BYOD archive must be a .zip file, got {{zip_name}}')\n"
                "    dataset_source = f'user upload: {{zip_name}}'\n"
                "else:\n"
                "    try:\n"
                "        req = urllib.request.Request(SAMPLE_DATASET_URL, headers={{'User-Agent': 'Mozilla/5.0'}})\n"
                "        with urllib.request.urlopen(req, timeout=30) as resp:\n"
                "            zip_bytes = resp.read()\n"
                "        actual_sha = hashlib.sha256(zip_bytes).hexdigest()\n"
                "        if actual_sha != SAMPLE_DATASET_SHA256:\n"
                "            raise ValueError(f'Sample dataset SHA-256 mismatch: expected {{SAMPLE_DATASET_SHA256}}, got {{actual_sha}}')\n"
                "        dataset_source = f'Cleanlab/cifar-10-subset (MIT license, sha256:{{actual_sha[:16]}}...)'\n"
                "    except Exception as exc:\n"
                "        print(f'Warning: public sample dataset download failed ({{exc}}); falling back to deterministic synthetic dataset.')\n"
                "        zip_bytes = None\n"
                "        dataset_source = 'synthetic stripes fallback'\n\n"
                "if zip_bytes is not None:\n"
                "    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:\n"
                "        # Security audit: reject directory traversal and absolute paths\n"
                "        for info in z.infolist():\n"
                "            if '..' in info.filename or info.filename.startswith(('/', '\\\\')):\n"
                "                raise ValueError(f'Security violation: illegal path in zip archive: {{info.filename}}')\n"
                "        names = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')) and not n.startswith('__MACOSX')]\n"
                "        if not names:\n"
                "            raise ValueError('No supported image files (.png, .jpg, .jpeg, .webp, .bmp) found in archive.')\n\n"
                "        has_train = any('train/' in n.lower() for n in names)\n"
                "        has_val = any('val/' in n.lower() or 'valid/' in n.lower() for n in names)\n\n"
                "        def _class_from_path(p):\n"
                "            parts = p.strip('/').split('/')\n"
                "            return parts[-2] if len(parts) >= 2 else 'unknown'\n\n"
                "        if has_train and has_val:\n"
                "            train_names = [n for n in names if 'train/' in n.lower()]\n"
                "            val_names = [n for n in names if 'val/' in n.lower() or 'valid/' in n.lower()]\n"
                "            CUSTOM_CLASSES = sorted(list({{_class_from_path(n) for n in train_names}}))\n"
                "            if len(CUSTOM_CLASSES) < 2:\n"
                "                raise ValueError(f'Classification requires at least 2 distinct classes, found {{CUSTOM_CLASSES}}')\n"
                "            cls_map = {{c: i for i, c in enumerate(CUSTOM_CLASSES)}}\n"
                "            for n in train_names:\n"
                "                cls = _class_from_path(n)\n"
                "                if cls in cls_map:\n"
                "                    train_images.append(Image.open(io.BytesIO(z.read(n))).convert('RGB'))\n"
                "                    train_targets.append(cls_map[cls])\n"
                "            for n in val_names:\n"
                "                cls = _class_from_path(n)\n"
                "                if cls in cls_map:\n"
                "                    val_images.append(Image.open(io.BytesIO(z.read(n))).convert('RGB'))\n"
                "                    val_targets.append(cls_map[cls])\n"
                "        else:\n"
                "            class_to_files = {{}}\n"
                "            for n in names:\n"
                "                cls = _class_from_path(n)\n"
                "                class_to_files.setdefault(cls, []).append(n)\n"
                "            CUSTOM_CLASSES = sorted(list(class_to_files.keys()))\n"
                "            if len(CUSTOM_CLASSES) < 2:\n"
                "                raise ValueError(f'Classification requires at least 2 distinct classes to train, found {{len(CUSTOM_CLASSES)}}: {{CUSTOM_CLASSES}}')\n"
                "            cls_map = {{c: i for i, c in enumerate(CUSTOM_CLASSES)}}\n"
                "            rng = random.Random(SEED)\n"
                "            for cls, files in class_to_files.items():\n"
                "                if len(files) < 2:\n"
                "                    raise ValueError(f'Class {{cls!r}} has fewer than 2 images ({{len(files)}}); cannot perform train/val split.')\n"
                "                f_list = list(files)\n"
                "                rng.shuffle(f_list)\n"
                "                if not USE_BYOD_DATASET and SUBSET_PER_CLASS:\n"
                "                    f_list = f_list[:SUBSET_PER_CLASS]\n"
                "                n_val = max(1, int(len(f_list) * VALIDATION_SPLIT))\n"
                "                val_files = f_list[:n_val]\n"
                "                train_files = f_list[n_val:]\n"
                "                for f in train_files:\n"
                "                    train_images.append(Image.open(io.BytesIO(z.read(f))).convert('RGB'))\n"
                "                    train_targets.append(cls_map[cls])\n"
                "                for f in val_files:\n"
                "                    val_images.append(Image.open(io.BytesIO(z.read(f))).convert('RGB'))\n"
                "                    val_targets.append(cls_map[cls])\n"
                "else:\n"
                "    CUSTOM_CLASSES = ['synthetic_horizontal_stripe', 'synthetic_vertical_stripe']\n"
                "    for cls_idx, pattern in enumerate(['horizontal', 'vertical']):\n"
                "        for i in range(6):\n"
                "            arr = np.zeros((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)\n"
                "            if pattern == 'horizontal':\n"
                "                arr[::16, :, 0] = 255\n"
                "                arr[:, :, 2] = (i * 30) % 255\n"
                "            else:\n"
                "                arr[:, ::16, 1] = 255\n"
                "                arr[:, :, 2] = (i * 30) % 255\n"
                "            img = Image.fromarray(arr, mode='RGB')\n"
                "            if i < 4:\n"
                "                train_images.append(img)\n"
                "                train_targets.append(cls_idx)\n"
                "            else:\n"
                "                val_images.append(img)\n"
                "                val_targets.append(cls_idx)\n\n"
                "ft_output_dir = 'outputs/{stem}_finetuned'\n"
                "fine_tuned_pipe, train_meta = pipe.fit(\n"
                "    train_images=train_images,\n"
                "    train_targets=train_targets,\n"
                "    val_images=val_images,\n"
                "    val_targets=val_targets,\n"
                "    class_names=CUSTOM_CLASSES,\n"
                "    epochs=1,\n"
                "    batch_size=4,\n"
                "    learning_rate=1e-4,\n"
                "    weights_dir=WEIGHTS_DIR,\n"
                "    output_dir=ft_output_dir,\n"
                ")\n"
                "print({{\n"
                "    'fine_tuning': 'complete',\n"
                "    'dataset_source': dataset_source,\n"
                "    'classes': CUSTOM_CLASSES,\n"
                "    'train_samples': len(train_images),\n"
                "    'val_samples': len(val_images),\n"
                "    'epochs': len(train_meta['history']),\n"
                "    'history': train_meta['history'],\n"
                "}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `dataset_source` (the pinned CIFAR-10 subset, or the synthetic fallback when offline), the class list, and the train / validation counts.\n\n<details><summary>Check '
                'your reasoning</summary>Usually, but read it rather than assume it: one epoch on 26 training images moves a new head only so far. Section 9 measures '
                'it against the majority-class baseline on held-out images. If the dataset download failed, the synthetic-stripes fallback is named in '
                '`dataset_source`.</details>'
            ),
        },
        {
            "md": (
                "## 9. Fresh-boundary reload and in-notebook evaluation\n\n"
                "To verify artifact integrity across an isolation boundary (simulating a fresh deployment or downstream "
                "container), `ResNet50ClassificationPipeline.from_pretrained` loads the newly generated `model.safetensors` "
                "and `model-config.json` directly from `outputs/{stem}_finetuned`. The pipeline re-instantiates the architecture, "
                "restores weights with `strict=True`, and configures the custom class labels. "
                "The reloaded pipeline is then evaluated across all held-out validation images (`val_images`) to compute "
                "top-1 validation accuracy, compare against the trivial majority-class baseline, and export machine-readable "
                "evaluation artifacts (`outputs/{stem}_validation_predictions.csv` and `outputs/{stem}_finetuned_evaluation_report.json`).\n\n"
                "*Evaluation practice.* **Predict before running:** on a balanced two-class split, what accuracy does the majority-class baseline get?"
            ),
            "code": (
                "import csv\n\n"
                "reloaded_pipe = ResNet50ClassificationPipeline.from_pretrained(weights_dir=ft_output_dir)\n\n"
                "# In-notebook evaluation across the held-out validation split\n"
                "val_labels = [CUSTOM_CLASSES[idx] for idx in val_targets]\n"
                "val_predictions = []\n"
                "val_hits = 0\n"
                "per_class_stats = {{cls_name: {{'total': 0, 'correct': 0}} for cls_name in CUSTOM_CLASSES}}\n\n"
                "for img, true_label in zip(val_images, val_labels):\n"
                "    pred_out = reloaded_pipe.predict(img)\n"
                "    pred = pred_out['predictions'][0]\n"
                "    is_correct = (pred['predicted_label'] == true_label)\n"
                "    if is_correct:\n"
                "        val_hits += 1\n"
                "    per_class_stats[true_label]['total'] += 1\n"
                "    if is_correct:\n"
                "        per_class_stats[true_label]['correct'] += 1\n"
                "    val_predictions.append({{\n"
                "        'true_label': true_label,\n"
                "        'predicted_label': pred['predicted_label'],\n"
                "        'predicted_index': pred['predicted_index'],\n"
                "        'confidence': pred['top_k'][0]['score'],\n"
                "        'correct': is_correct,\n"
                "    }})\n\n"
                "val_accuracy = val_hits / max(len(val_labels), 1)\n"
                "majority_class = max(set(val_labels), key=val_labels.count)\n"
                "majority_baseline = val_labels.count(majority_class) / max(len(val_labels), 1)\n\n"
                "finetuned_eval_report = {{\n"
                "    'task': 'single-label classification fine-tuning evaluation',\n"
                "    'verdict': 'success',\n"
                "    'validation_samples': len(val_labels),\n"
                "    'classes': CUSTOM_CLASSES,\n"
                "    'metrics': {{\n"
                "        'accuracy': round(val_accuracy, 4),\n"
                "        'majority_baseline_accuracy': round(majority_baseline, 4),\n"
                "        'accuracy_lift_over_baseline': round(val_accuracy - majority_baseline, 4),\n"
                "        'mean_confidence': round(sum(p['confidence'] for p in val_predictions) / max(len(val_predictions), 1), 4),\n"
                "    }},\n"
                "    'per_class_breakdown': {{\n"
                "        c: {{\n"
                "            'total': s['total'],\n"
                "            'correct': s['correct'],\n"
                "            'accuracy': round(s['correct'] / s['total'], 4) if s['total'] > 0 else 0.0,\n"
                "        }} for c, s in per_class_stats.items()\n"
                "    }},\n"
                "    'fresh_boundary_source': reloaded_pipe.source,\n"
                "}}\n\n"
                "eval_report_path = f'outputs/{stem}_finetuned_evaluation_report.json'\n"
                "with open(eval_report_path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(finetuned_eval_report, handle, indent=2, ensure_ascii=False)\n\n"
                "val_preds_csv_path = f'outputs/{stem}_validation_predictions.csv'\n"
                "with open(val_preds_csv_path, 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['sample_index', 'true_label', 'predicted_label', 'correct', 'confidence'])\n"
                "    writer.writeheader()\n"
                "    for idx, p in enumerate(val_predictions):\n"
                "        writer.writerow({{'sample_index': idx, 'true_label': p['true_label'], 'predicted_label': p['predicted_label'], 'correct': p['correct'], 'confidence': f\"{{p['confidence']:.6f}}\"}})\n\n"
                "print('=== Fine-Tuned Model In-Notebook Evaluation ===')\n"
                "print(f\"Validation Samples: {{len(val_labels)}} | Classes: {{CUSTOM_CLASSES}}\")\n"
                "print(f\"Top-1 Accuracy:     {{val_accuracy:.2%}} ({{val_hits}}/{{len(val_labels)}} correct)\")\n"
                "print(f\"Majority Baseline:  {{majority_baseline:.2%}} (predicting '{{majority_class}}')\")\n"
                "print(f\"Lift Over Baseline: {{val_accuracy - majority_baseline:+.2%}}\")\n"
                "print(json.dumps(finetuned_eval_report['metrics'], indent=2))\n"
            ),
        },
        {
            "md": (
                '**What to notice:** top-1 accuracy, the majority baseline, the lift, the per-class breakdown, and `fresh_boundary_source`.\n\n<details><summary>Check '
                'your reasoning</summary>Half: with two balanced classes, always answering one class is right 50 % of the time. Only the lift over that baseline says '
                'the head learned something. A handful of validation images gives no dispersion estimate, and the reload from `outputs/` is the evidence that the '
                'exported files, not the in-memory model, produce these predictions.</details>'
            ),
        },
        {
            "md": (
                "## 10. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full prediction (argmax decision and the rank-ordered top-k scores), "
                "the evaluation report, the fine-tuning training history and in-notebook evaluation metrics, the sample identity and digest, "
                "the notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, and the runtime identity (Python, `torch`, `timm`, device). The rank-ordered top-k table and fine-tuned validation "
                "predictions are written as CSV files so class ordering survives downstream use. Deployable fine-tuned model artifacts and evaluation "
                "reports are published in `outputs/{stem}_finetuned/` and `outputs/`. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'fine_tuning': {{\n"
                "        'history': train_meta['history'],\n"
                "        'classes': CUSTOM_CLASSES,\n"
                "        'evaluation': finetuned_eval_report,\n"
                "    }},\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'timm': timm.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_top_k.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'index', 'label', 'score'])\n"
                "    for rank, item in enumerate(prediction['top_k'], start=1):\n"
                "        writer.writerow([image_name, rank, item['index'], item['label'], f\"{{item['score']:.6f}}\"])\n"
                "print({{'outputs': sorted(os.listdir('outputs')), 'finetuned': sorted(os.listdir(ft_output_dir))}})\n"
                "print(['outputs/{stem}_finetuned/model.safetensors', 'outputs/{stem}_finetuned/model-config.json'])"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The predicted label is the argmax of a softmax over the fixed 1000-class ImageNet-1k label space (or custom "
        "class names when fine-tuned); the `score` values are uncalibrated softmax outputs, not probabilities of "
        "correctness, and the pipeline ships no threshold. On the synthetic gradient the label is meaningless by "
        "construction and the evaluation report says `not-measurable`; a `top_k_accuracy` value shown for a single "
        "BYOD image is tutorial evidence for that one image and must not be generalized to a domain, camera, or class "
        "distribution. Images whose subject is outside ImageNet-1k, line drawings, medical or satellite imagery, and "
        "subjects near the image border (removed by the center crop) all degrade results in ways the pipeline does not "
        "detect. The pipeline provides no detection, segmentation, multi-label, OCR, or open-vocabulary capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, perform in-kernel fine-tuning, reload the verified artifact bundle, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a photograph of a known ImageNet class and its index to see the "
        "report switch to `sample-sanity` with `top_k_accuracy` at k=1 and k=5; supply your own multi-class dataset folder "
        "to `pipe.fit` to adapt to your domain; classify a batch (up to `MAX_BATCH`) of labelled images from your own "
        "domain and compare top-1 against the majority-class baseline of that set.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 8 prints "falling back to deterministic '
        'synthetic dataset"** — the pinned CIFAR-10 subset could not be downloaded or failed its SHA-256; the run continues on synthetic stripes and says so in '
        '`dataset_source`. Re-run Section 8 when the network is back.\n- **BYOD image: "Pillow cannot decode this file"** — the file is not an image Pillow '
        'reads (PNG, JPEG, WebP and similar).\n- **BYOD dataset: a class or path refusal** — at least two class folders with at least two images each; no `..` '
        'or absolute paths in the archive.\n- **BYOD: "… is not a file"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the '
        'upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the file in the runtime and set `BYOD_IMAGE_PATH` or `BYOD_DATASET_PATH`.\n- '
        '**BYOD: "Upload exactly one …"** — the dialog was cancelled or several files were chosen; run the cell again.\n\n## Glossary\n\n- **ImageNet-1k:** the '
        '1000-class label space the pretrained head predicts.\n- **Logits / softmax score:** the raw class outputs and their normalised scores; uncalibrated, '
        'relative to the 1000 classes, not probabilities of being right.\n- **Argmax / top-k:** the highest-scoring class; the k highest, in descending order.\n- '
        "**Input manifest:** the record of what was validated, under which ceilings, and any rejections.\n- **`not-measurable` / `sample-sanity`:** the report's "
        'verdict without a label; with a label on a tutorial-sized sample.\n- **Backbone / head:** the feature extractor kept from pretraining; the final linear '
        'layer, replaced and trained for your classes.\n- **Majority-class baseline:** always answering the most frequent class; accuracy must beat it to mean anything.\n- '
        '**Fresh-boundary reload:** reloading the exported files from disk into a new pipeline, so the evaluation uses the artifact, not memory.\n- **Isolated '
        'environment:** the separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\n- On the '
        'synthetic gradient the model answered ___ with top-1 score ___, which I read as ___.\n- After head fine-tuning the reloaded model scored ___ on the '
        'held-out split against a majority baseline of ___.\n- The number I would not trust on its own is ___, because ___.\n- Before using this on my own images '
        'I would check ___ and compare against ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/resnet50-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/huggingface/pytorch-image-models\n"
        "- ResNet strikes back (A1 training recipe): https://arxiv.org/abs/2110.00476\n"
        "- Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385"
    ),
}
