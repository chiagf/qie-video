"""qie-video - Wan2.2 I2V: first/last frame in, an mp4 out.

Served by uvicorn on a rented GPU box (Vast/RunPod). The Gradio UI is mounted into FastAPI so
`/health`, `/status` and `/video-config` sit beside it. No ZeroGPU booking: the engine owns the
card outright.

The engine comes from the `qie_engine` submodule, so this app never vendors a second copy of the
Wan stack.
"""

from __future__ import annotations

import base64
import os
import secrets
import time
import traceback

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import prompting
import video_ui
from config import CONFIG
from qie_engine import WanEngine, gpu_info, wan_memory_plan
from store import ClipStore

BOOT_ERROR = ""
BOOTED_AT = time.time()

engine = WanEngine(CONFIG.wan)
store = ClipStore(CONFIG.out_dir, CONFIG.max_history)

# Load before serving: the weights are ~45 GB and the first request would otherwise time out.
if CONFIG.preload:
    BOOT_ERROR = engine.load() or ""


app = FastAPI(title="qie-video")

if CONFIG.auth and ":" in CONFIG.auth:
    _user, _, _password = CONFIG.auth.partition(":")
    _expected = f"{_user}:{_password}".encode()

    @app.middleware("http")
    async def _basic_auth(request: Request, call_next):
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                if secrets.compare_digest(base64.b64decode(header[6:]).strip(), _expected):
                    return await call_next(request)
            except Exception:  # noqa: BLE001 - a malformed header is just a failed login
                pass
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="qie-video"'})


@app.get("/health")
def health():
    return {
        "ok": engine.loaded,
        "mode": CONFIG.mode,
        "boot_error": BOOT_ERROR,
        "uptime_s": round(time.time() - BOOTED_AT, 1),
        "session": len(store),
    }


@app.get("/status")
def status():
    return {
        "gpu": gpu_info(),
        "engine": engine.stats(),
        "session": len(store),
        "rife": engine.rife.stats(),
    }


@app.get("/video-config")
def video_config():
    info = gpu_info()
    plan = wan_memory_plan(
        CONFIG.wan.dtype, engine.offload or CONFIG.wan.offload, info.get("total_gb", 0.0)
    )
    return {
        "app": "qie-video",
        "mode": CONFIG.mode,
        "gpu": info,
        "engine": engine.stats(),
        "memory": plan,
        "config": CONFIG.public(),
        "vlm": {
            "enabled": CONFIG.vlm_enabled,
            "provider": CONFIG.vlm_provider,
            "model": CONFIG.vlm_model,
            "token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        },
        "presets": [label for label, _ in prompting.PRESETS],
    }


demo = video_ui.build(engine, store, CONFIG)
# SSR off: Gradio 6 otherwise starts its Node front proxy on the user-facing port and moves
# Python behind it, which makes the uvicorn bind fail with EADDRINUSE.
app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft(), ssr_mode=False)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    print(
        f"[qie-video] starting on :{port} mode={CONFIG.mode} preload={CONFIG.preload} "
        f"dtype={CONFIG.wan.dtype} offload={CONFIG.wan.offload} "
        f"scheduler={CONFIG.wan.scheduler} fluidity={CONFIG.wan.fluidity}",
        flush=True,
    )
    if BOOT_ERROR:
        print(f"[qie-video] boot error: {BOOT_ERROR}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
