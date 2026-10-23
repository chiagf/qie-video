# qie-video

**Wan2.2 I2V-A14B** in a container: first frame (optionally a last frame) in, an mp4 out.
Runs on a rented GPU box - Vast or RunPod - not as an HF Space.

The Wan engine lives in the [`qie-engine`](https://github.com/chiagf/qie-engine) submodule, so
this app owns the *UI* and the engine is shared. UI features come from
[`gchia/wan777-Video`](https://huggingface.co/spaces/gchia/wan777-Video) (attribution below),
ported off ZeroGPU onto a portable path.

## Features

- **First + optional last frame** - image-to-video, or a guided transition between two frames.
- **Scheduler + flow shift** - FlowMatchEuler (default, `shift=8.0`), UniPC, DEIS, DPM++.
- **RIFE fluidity** - render a 16 fps clip at 32/64/128 fps by optical-flow interpolation.
  Falls back to linear cross-fade if the model cannot load; the status line says which ran.
- **Use current frame as input** - grab the playing clip's timestamp in the browser, pull that
  frame out of the mp4, feed it back in as the next first frame.
- **Next Segment** - a vision model reads the last frame and writes the next clip's motion, so
  you can chain clips without retyping the prompt.
- **Rewrite** - a vision model makes your instruction specific about camera path, action and
  pacing while keeping the subject's identity stable.

## Run on Vast / RunPod

Pull the image:

```
ghcr.io/chiagf/qie-video:latest
```

Instance: **RTX 4090 48 GB** (or any CUDA card with ~48 GB), `--disk >= 50` (60 recommended).
Port **7860**.

Environment:

| Var | Value | |
| --- | --- | --- |
| `HF_TOKEN` | `hf_...` | optional; speeds the weight download, enables Rewrite/Next Segment |
| `QIE_AUTH` | `user:pass` | optional; HTTP Basic gate over every route |
| `QIE_WAN_OFFLOAD` | `model_cpu` | optional; headroom if 480p ever gets tight |
| `QIE_WAN_STACK` | `download` | default; `baked` if you built a weights image |

Weights download on first boot into `/data/hf` (~45 GB). Give the instance 60 GB of disk.

### What fits

At the default 480x832 / ~81 frames the denoise peak is **~33 GiB**, so a 48 GB card runs it
without offloading. 720x1280 at 81 frames needs `QIE_WAN_OFFLOAD=model_cpu`.

### Check it came up

```
curl http://<host>:7860/health      # {"ok": true, "mode": "video", ...}
curl http://<host>:7860/status      # gpu + engine + rife
curl http://<host>:7860/video-config
```

## Env vars

| Var | Default | |
| --- | --- | --- |
| `QIE_WAN_DTYPE` | `fp8` | `fp8` \| `bf16` |
| `QIE_WAN_OFFLOAD` | `auto` | `auto` \| `model_cpu` \| `sequential` \| `none` |
| `QIE_WAN_STEPS` | `8` | Lightning is tuned for 4-8 |
| `QIE_WAN_SHIFT` | `8.0` | FlowMatchEuler shift |
| `QIE_WAN_MAX_MP` | `0.40` | output ceiling (480x832) |
| `QIE_WAN_SCHEDULER` | `FlowMatchEulerDiscrete` | see `qie_engine.settings.SCHEDULERS` |
| `QIE_WAN_FLUIDITY` | `1` | `1` \| `2` \| `4` \| `8` (RIFE) |
| `QIE_WAN_RIFE_ENABLED` | `on` | off = never interpolate |
| `QIE_WAN_LORA_ENABLED` | `on` | lightx2v Lightning pair |
| `QIE_VLM` | `on` | `off` = template prompts only |
| `QIE_VLM_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` | vision model for Rewrite / Next Segment |

## Build

```
docker build -t qie-video .
docker run --gpus all -p 7860:7860 -v qie-data:/data -e HF_TOKEN=hf_xxx qie-video
```

The `qie_engine` submodule must be checked out (`git clone --recurse-submodules`, or
`git submodule update --init`). The Dockerfile pip-installs it into the image.

## Tests

CPU-only, no torch, no network:

```
python -m unittest discover -s tests -v
```

The engine's own tests live in the submodule.

## Attribution

UI structure and the frame-grab / RIFE / scheduler features are adapted from
[`gchia/wan777-Video`](https://huggingface.co/spaces/gchia/wan777-Video), itself a community
Wan2.2 Lightning Space. RIFE inference comes from
[r3gm/RIFE](https://huggingface.co/r3gm/RIFE) (Apache-2.0, originally megvii-research/RIFE);
the `model/` support code it imports is vendored inside `qie-engine`, trimmed to the inference
path. Wan2.2 weights are under their own licenses on Hugging Face - check them before
commercial use.
