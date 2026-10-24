# Pack A: MOSS Control

One node: **AIRPG MOSS Control (adapter)** (`AIRPG/moss`).

Loads a MOSS control adapter (a single `.safetensors` holding LoRA weights plus a control embedder) from your `loras` folder and applies it to a MOSS-Image model. Wire in any of `image`, `mask`, `lineart`, `depth`, `colour`, `pose`, `tile` to steer generation. Outputs the patched `model` and an empty `latent` of the right size.

- `strength` 0 (or nothing connected) = plain MOSS-Image; the base model is never replaced.
- Needs an adapter file in the `moss-control-adapter-v1` format. The training script isn't included.
- Core logic is dependency-free torch, with unit tests in `tests/`.

Install: copy `comfyui-airpg-moss-control` into `ComfyUI/custom_nodes`.
