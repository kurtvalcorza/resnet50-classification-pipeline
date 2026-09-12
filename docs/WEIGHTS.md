# Weight provenance and DIMER hosting

- Upstream: `timm/resnet50.a1_in1k`
- Immutable revision: `767268603ca0cb0bfe326fa87277f19c419566ef`
- Weight format: SafeTensors (`model.safetensors`, 102469840 bytes)
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/resnet50-a1/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 102509031); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/resnet50_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `timm==1.0.29` built-in `resnet50` architecture; weights loaded from a file path via `pretrained_cfg_overlay`; no remote code is executed. Hub download is opt-in and pinned to the revision above.
