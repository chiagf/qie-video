# Baking the weights into the image — qie-video

Companion to `README.md` §Build. Builds `ghcr.io/chiagf/qie-video:weights` (the two Wan2.2
I2V-A14B fp8 experts + the Lightning LoRA, ~31 GB) and then `:baked`, the app image layered on it.
Booting with `QIE_WAN_STACK=baked` then skips the ~28 GB Comfy-Org download at first boot.

## 1. Why / when

Optional — the default image works, it just downloads ~28 GB of experts into `/data/hf` on the first
boot of every new instance. Bake when you rent often enough to care, or when the instance's disk
would rather hold outputs than a second copy of weights you already have.

**This is a local build, not a CI build.** BuildKit holds each layer uncompressed *and* compressed
while pushing, so a 31 GB image needs ~62 GB of free disk; a free GitHub runner has ~44 GB. GitHub
Actions therefore cannot produce this image — you build it on your own machine (or a rented box).

## 2. What your PC needs

| | |
| --- | --- |
| Docker Desktop for Windows, WSL2 backend | ~2-3 GB + **one reboot** |
| Settings → Resources → Advanced → **Disk image size** | **150 GB** (the default is too small) |
| Free physical space on the drive holding that VHDX | 150 GB |
| Peak inside the VHDX | ~80-90 GB (weights image + push temp + app layer + build cache) |
| Download | ~38 GB (30 GB HF weights, ~7 GB pip/torch, base images) |
| Upload | ~43 GB (31 GB for `:weights`, then only ~12 GB for `:baked` — GHCR already has the base layers) |
| Time | 1.5-3 h, almost all of it your uplink |

Nothing installs into your Windows Python, and no weights are written outside Docker — the bake runs
inside the build container.

**The VHDX does not shrink** when you later delete images; compaction is a separate manual step.

## 3. Phase A — setup (once)

1. Install **Docker Desktop for Windows** from <https://www.docker.com/products/docker-desktop/>,
   WSL2 backend. It asks for a **reboot**.
2. Start it → **Settings → Resources → Advanced** → **Disk image size = 150 GB**, location on a drive
   with 150 GB free → **Apply & restart**.
3. ```
   docker --version
   ```
4. GitHub → Settings → Developer settings → **Fine-grained tokens** → repository access: only
   `chiagf/qie-video` → Permissions → **Packages: Read and write** → generate.
5. ```
   docker login ghcr.io
   ```

No Hugging Face token is needed for the bake: `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` and
`lightx2v/Wan2.2-Lightning` are public. Your shell already exports `HF_TOKEN` if HF ever rate-limits
you during the download.

## 4. Phase B — three changes in the repo

1. From `C:\_UPlus.Dev\H3-RunPod`:

   ```powershell
   Copy-Item qie-studio\bake_weights.py    qie-video\bake_weights.py
   Copy-Item qie-studio\Dockerfile.weights qie-video\Dockerfile.weights
   ```

   Both are drop-in: `bake_weights.py` only touches those two HF repos and `Path("/models")`, and the
   `Dockerfile.weights` instructions mention `qie-studio` only in comments.
2. *(cosmetic)* In `qie-video\Dockerfile.weights`, change `qie-studio` → `qie-video` and
   `config.py` → `qie_engine/settings.py` in the header comment. The 11 instruction lines are correct
   as they stand.
3. In `qie-video\Dockerfile`, replace lines 14-15 — keep everything above line 14, and keep
   `# syntax=` on the first line:

   ```dockerfile
   # syntax=docker/dockerfile:1.7
   ARG BASE_IMAGE=python:3.10-slim-bookworm
   FROM ${BASE_IMAGE}
   ```

   The default preserves plain `docker build -t qie-video .` and CI.

**The LoRA needs one more flag.** `QIE_WAN_STACK=baked` rewrites the expert and pipeline paths but
not `lora_repo` (`qie_engine/settings.py`), so `/models/lora/...` is only read if you point at it:

- **Recommended, no code:** always pass `-e QIE_WAN_LORA=/models/lora` (Phase D). The default
  `lora_subdir` already matches the baked layout, and `wan.py` passes it as `subfolder`.
- Alternative, a code fix: rewrite `lora_repo` to `/models/lora` when baked, inside
  `qie_engine/settings.load()`. That is a commit in the `qie-engine` submodule plus a pin bump here.

`_lora()` is best-effort (`wan.py:211-247`): on failure it only prints
`[wan] high-noise Lightning LoRA not loaded (...)` and `self.lora` stays empty. The app still starts —
it just runs 8 steps without the 4-step turbo. The `/status` check in Phase D is what catches that.

## 5. Phase C — build and push

```powershell
cd C:\_UPlus.Dev\H3-RunPod\qie-video

# 1. weights image: downloads ~30 GB from HF, 10-30 min
docker build --progress=plain -f Dockerfile.weights -t ghcr.io/chiagf/qie-video:weights .

# 2. push it: ~31 GB up (needs ~62 GB free inside the VHDX)
docker push ghcr.io/chiagf/qie-video:weights

# 3. app image on top: pip-installs torch/diffusers, 5-15 min
docker build --build-arg BASE_IMAGE=ghcr.io/chiagf/qie-video:weights -t ghcr.io/chiagf/qie-video:baked .

# 4. push it: only the ~12 GB of new layers, GHCR already has the base layers
docker push ghcr.io/chiagf/qie-video:baked
```

Confirm the bake landed before pushing step 2:

```
docker run --rm ghcr.io/chiagf/qie-video:weights ls -lh /models
```

You should see `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`,
`wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, and under
`lora/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/` both `high_noise_model.safetensors` and
`low_noise_model.safetensors`.

`:weights` is a base image only — it has no app in it. `:baked` is the one you rent.

## 6. Phase D — run it

Add the two flags to the `env` string in `create_vast_video_template.ps1`, then re-run it with
`-Tag baked` and a real `-Auth` (replacing `CHANGEME`):

```
-e QIE_WAN_STACK=baked -e QIE_WAN_LORA=/models/lora
```

Keep `-Disk 60`: the bake is **partial** — the umt5/VAE/CLIP-vision half (~13.5 GB) and the 22 MB
RIFE archive still download into `/data/hf` on first boot. Only the two experts and the LoRA are in
the image.

Checks once it is up:

1. Boot log shows **no** `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` download — only
   `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (~13.5 GB). Correct for a partial bake.
2. `GET /status` → `engine.lora` must read **`/models/lora [lightning_high + lightning_low]`**.
   An empty string means the LoRA did not load (missing `QIE_WAN_LORA`, see Phase B).
3. `GET /health` → `"ok": true`.
4. Render one 720p clip.

Locally, without renting:

```
docker run --gpus all -p 7860:7860 -e QIE_WAN_STACK=baked -e QIE_WAN_LORA=/models/lora -e QIE_AUTH=user:pass ghcr.io/chiagf/qie-video:baked
```

## 7. Why the bake must run inside Docker

`bake_weights.py` writes to `Path("/models")`. Run it in PowerShell and that is `C:\models` — a
folder on your Windows disk that no Docker build can ever see, so you would have downloaded 30 GB
for nothing.

You never execute the script yourself. `docker build -f Dockerfile.weights ... .` sends it to
Docker, and the `RUN python /bake_weights.py` line runs it inside an ephemeral **Linux** container
where `/models` is a real absolute path — so the weights become a layer of the image. The trailing
`rm -rf /bake_weights.py /tmp/hf` drops the script and the download cache.

So the only supported way to run it is `docker build -f Dockerfile.weights` (Phase C, step 1).

## 8. Gotchas

- **`/models` must stay outside `/data`.** Vast mounts its instance volume at `/data` by default,
  which would shadow anything baked there. The bake script uses `/models` for exactly that reason;
  don't move it onto the volume.
- **Never `python bake_weights.py` on Windows** — see §7.
- **CI cannot build this image** (44 GB runner vs ~62 GB needed). Producing it in CI would need a
  larger runner or a rented build box.
- **The VHDX grows and never auto-shrinks.** Deleting `:weights` reclaims space inside the VHDX, not
  on the host, until you compact it.
- **`:weights` and `:baked` are additive.** CI keeps publishing `:<sha>` and `:latest` untouched;
  the default `docker build` still works because `BASE_IMAGE` defaults to `python:3.10-slim-bookworm`.
- **Partial vs full bake.** Adding the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` pipeline to
  `/models/pipeline` would give a ~44 GB image and a zero-download boot — `qie_engine/settings.py`
  already supports it when `model_index.json` is present — but the push then needs ~90 GB free and
  every Vast pull costs ~13 GB more. Not covered here on purpose.
