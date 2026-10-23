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

Instance: **RTX 4090 48 GB** (or any CUDA card with ~48 GB), `--disk >= 60`.
Port **7860**.

Environment:

| Var | Value | |
| --- | --- | --- |
| `HF_TOKEN` | `hf_...` | optional; speeds the weight download, enables Rewrite/Next Segment |
| `QIE_AUTH` | `user:pass` | optional; HTTP Basic gate over every route |
| `QIE_WAN_OFFLOAD` | `model_cpu` | optional; headroom if 480p ever gets tight |
| `QIE_WAN_STACK` | `download` | default; `baked` if you built a weights image |

Weights download on first boot into `/data/hf` (~45 GB). Give the instance 60 GB of disk.

### Rent from the saved Vast template

A private template on this account already pins the image, port, auth and offer filter:

| | |
| --- | --- |
| name | `QIE Video bb05d82` |
| id | `776154` |
| hash | `8952e974d4032724394895c47915e69b` |
| disk | 60 GB |
| env | `-e OPEN_BUTTON_PORT=7860 -e QIE_AUTH=vast:CHANGEME -p 7860:7860` |

Find an offer with the template's own filter, then rent against the hash:

```
vastai search offers 'compute_cap>=890 cpu_arch in ["amd64"] gpu_ram>=40 gpu_total_ram>=40 disk_space>=60 num_gpus in ["1"] rentable=true inet_down>=500' -o dph
vastai create instance <OFFER_ID> --template_hash 8952e974d4032724394895c47915e69b --disk 60
```

Replace the password when you rent (`--env` replaces the template's, so pass all three):

```
vastai create instance <OFFER_ID> --template_hash 8952e974d4032724394895c47915e69b --disk 60 --env '-e OPEN_BUTTON_PORT=7860 -e QIE_AUTH=vast:REALPASS -p 7860:7860'
```

`HF_TOKEN` is **not** in the template (templates are shareable). It already exists at the account
level, so every instance picks it up: `vastai show env-vars`.

Filter notes, all verified against the live marketplace:

- `compute_cap>=1200` (the longcat default) is Blackwell-only and **excludes every RTX 4090**
  (4090 = sm_89 = compute_cap 890). `>=890` is Ada-and-newer, which includes it and gives native fp8.
- `gpu_ram` is compared in **gigabytes** - `gpu_ram>=40000` matches nothing, `gpu_ram>=40` is right.
- There is **no working host-RAM filter** (`ram>=64`, `ram>=65536`, `cpu_ram>=65536` all return 0).
  Read the RAM column instead; the median on this filter is ~128 GB.
- `disk_space>=60` matters: without it the cheapest offer had only 36.6 GB free, which cannot hold
  the ~45 GB of weights. With it, every offer has >= 87 GB.

Rebuild the template whenever the image pin moves (the script lives one level up, next to the
longcat one, and refuses to save a filter that matches 0 offers):

```
.\create_vast_video_template.ps1 -Tag <new-sha>
```

### What fits

At the default 480x832 / ~81 frames the denoise peak is **~33 GiB**, so a 48 GB card runs it
without offloading. 720x1280 at 81 frames needs `QIE_WAN_OFFLOAD=model_cpu`.

Host RAM matters too: building both experts at load peaks near **48 GB** (fp8). Rent a box with
**64 GB or more**; a host below the limit now refuses to start with a readable error instead of
being silently OOM-killed.

### Check it came up

```
curl http://<host>:7860/health      # {"ok": true, "state": "ready", "attempts": 1, ...}
curl http://<host>:7860/status      # gpu + engine + rife
curl http://<host>:7860/video-config
```

The server answers `/health` from the first second, while the ~45 GB download is still
running: `state` is `loading` until the weights land, then `ready`. `/health` is the one route
exempt from `QIE_AUTH` (it reports state only, no secrets) so this command works with the gate on.

If a load attempt fails (network blip, disk too small) the log shows
`load attempt N failed: ... - retrying in 15s` and it keeps trying every 15-120 s.
**Restart the container in place, don't recreate the instance** - `/data/hf` survives a restart
and resumes the download; a fresh instance starts the 45 GB over. The last real reason is in
`/health` as `boot_error`.

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
| `QIE_WAN_STACK` | `download` | `baked` = load from `/models` in a weights-baked image |
| `QIE_WAN_LORA` | `lightx2v/Wan2.2-Lightning` | HF repo, or `/models/lora` when `QIE_WAN_STACK=baked` |
| `QIE_VLM` | `on` | `off` = template prompts only |
| `QIE_VLM_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` | vision model for Rewrite / Next Segment |

## Build

```
docker build -t qie-video .
docker run --gpus all -p 7860:7860 -v qie-data:/data -e HF_TOKEN=hf_xxx qie-video
```

The `qie_engine` submodule must be checked out (`git clone --recurse-submodules`, or
`git submodule update --init`). The Dockerfile pip-installs it into the image.

Weights are left out of that build: the first boot downloads ~28 GB of experts into `/data/hf`. To
bake them into the image instead — and skip that download — follow
[BAKE-WEIGHTS.md](BAKE-WEIGHTS.md), which builds a `:weights` base image plus a `:baked` app image.
That build is too big for GitHub Actions (~62 GB needed, a free runner has ~44 GB), so it runs on
your own machine.

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
