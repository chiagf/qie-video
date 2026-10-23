"""The qie-video UI: two frames in, one Wan2.2 clip out, with the wan777 feature set.

First + optional last frame, scheduler + flow-shift, RIFE fluidity, quality, and the "use
current frame as input" grab (a JS snippet reads the player timestamp, Python pulls that frame
out of the mp4 with cv2). Built by `build()` and mounted into the same FastAPI app as the
health routes, so `/health` and `/video-config` keep working.

Vast/rented only: no ZeroGPU booking, no `spaces` import.
"""

from __future__ import annotations

import time

import gradio as gr

import prompting
from config import Config
from qie_engine.settings import FLUIDITY_CHOICES, MAX_SEED, SCHEDULERS, WAN_NEGATIVE
from prompting import PRESETS

# Reads the playing clip's timestamp. Runs in the browser, so it cannot touch Python state -
# it only fills a hidden Number that a `change` handler then feeds to extract_frame().
GRAB_TIMESTAMP_JS = """
function() {
    const video = document.querySelector('#generated-video video');
    return video ? video.currentTime : 0;
}
"""


def extract_frame(video_path: str, timestamp: float):
    """Pull one frame out of the rendered clip at `timestamp` seconds."""
    if not video_path:
        return None
    try:
        import cv2
        from PIL import Image
    except Exception:  # noqa: BLE001 - opencv is in the image, but never assume
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = int(float(timestamp) * fps)
    if total > 0:
        index = max(0, min(index, total - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def last_frame_of(video_path: str):
    """The final frame of a clip - the natural input for the next segment."""
    if not video_path:
        return None
    try:
        import cv2
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = capture.read()
    capture.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None


def _pick_seed(seed, randomize: bool) -> int:
    import random

    if randomize or seed is None or int(seed) < 0:
        return random.randint(0, MAX_SEED)
    return int(seed) % MAX_SEED


def _status_line(result, total_s: float) -> str:
    extra = ""
    if result.interpolated:
        extra += f" · interpolated `{result.interpolated}`"
    if result.scheduler:
        extra += f" · scheduler `{result.scheduler}`"
    return (
        f"**{result.width} x {result.height}** · {result.frames} frames @{result.fps} fps"
        f" · render {result.ms / 1000:.1f}s · peak {result.peak_gb} GiB"
        f" · total {total_s:.1f}s · seed `{result.seed}`{extra}"
    )


def _paths(store) -> list[str]:
    return [entry.path for entry in store.items()]


def _boot_banner(engine) -> str:
    """Live boot state for the header. The old static check ran once at build time, before the
    background loader had produced any error to show."""
    if engine.loaded:
        return ""
    if engine.error:
        return f"**model error** (retrying in the background): `{engine.error}`"
    return (
        "**model loading** - the first boot downloads ~45 GB into /data/hf; "
        "Generate is disabled until it finishes."
    )


def _render(engine, store, cfg, first, last, prompt, duration, steps, guidance, guidance_2,
            negative, seed, randomize, scheduler, flow_shift, fluidity, quality,
            progress=gr.Progress()):
    """The GPU half: validate, sample, register the clip. Nothing else runs on the worker."""
    if not engine.loaded:
        raise gr.Error(f"video model not ready: {engine.error or 'still loading'}")
    first_path = first if isinstance(first, str) else None
    if not first_path:
        raise gr.Error("upload a first frame")
    last_path = last if isinstance(last, str) else None
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise gr.Error("write what happens in the clip")

    seed = _pick_seed(seed, randomize)
    started = time.time()
    try:
        result = engine.run(
            first_path,
            last_path,
            prompt_text,
            duration=duration,
            steps=int(steps),
            seed=seed,
            guidance=guidance,
            guidance_2=guidance_2,
            negative=negative,
            scheduler=scheduler,
            flow_shift=flow_shift,
            fluidity=int(fluidity),
            quality=int(quality),
            progress=progress,
            out_dir=cfg.out_dir,
        )
    except Exception as error:  # noqa: BLE001 - the UI shows the real reason
        raise gr.Error(f"{type(error).__name__}: {error}") from error
    store.register_clip(
        result.path,
        name=f"clip-{result.seed}.mp4",
        prompt=prompt_text,
        seed=result.seed,
        width=result.width,
        height=result.height,
        frames=result.frames,
        fps=result.fps,
        ms=result.ms,
    )
    return result.path, _paths(store), _status_line(result, time.time() - started)


def build(engine, store, cfg: Config) -> gr.Blocks:
    with gr.Blocks(title="QIE Video") as demo:
        gr.Markdown(
            "# QIE Video\n"
            f"**Wan2.2 I2V-A14B** ({cfg.wan.dtype}, {cfg.wan.attn} attention,"
            f" {cfg.wan.scheduler} scheduler). Upload a frame to start from (optionally one to"
            " end on), describe the motion, get an mp4. Click a clip in Session to keep going."
        )
        boot_status = gr.Markdown(_boot_banner(engine))
        demo.load(lambda: _boot_banner(engine), None, boot_status)

        with gr.Row():
            with gr.Column(scale=3):
                first = gr.Image(label="First frame", type="filepath", sources=["upload", "clipboard"])
                last = gr.Image(label="Last frame (optional)", type="filepath", sources=["upload", "clipboard"])
                with gr.Row():
                    grab_btn = gr.Button("Use current frame as input", variant="secondary")
                    last_btn = gr.Button("Use latest clip's last frame", variant="secondary")
                    timestamp = gr.Number(value=0, visible=False, elem_id="hidden-timestamp")

                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="slow dolly in while the subject turns; the light holds. cinematic, 35mm.",
                )
                with gr.Row():
                    rewrite_btn = gr.Button("Rewrite", variant="secondary")
                    segment_btn = gr.Button("Next Segment", variant="secondary")
                    preset = gr.Dropdown(
                        [label for label, _ in PRESETS], label="Preset", value=None, scale=1
                    )
                prompt_status = gr.Markdown("")

                with gr.Accordion("Settings", open=False):
                    duration = gr.Slider(0.5, 10.0, value=4.0, step=0.1, label="Duration (seconds)")
                    steps = gr.Slider(1, 30, value=cfg.wan.steps, step=1, label="Steps")
                    guidance = gr.Slider(0.0, 10.0, value=cfg.wan.guidance, step=0.5,
                                         label="Guidance (high-noise expert)")
                    guidance_2 = gr.Slider(0.0, 10.0, value=cfg.wan.guidance_2, step=0.5,
                                           label="Guidance 2 (low-noise expert)")
                    scheduler = gr.Dropdown(list(SCHEDULERS), value=cfg.wan.scheduler, label="Scheduler")
                    flow_shift = gr.Slider(0.5, 15.0, value=cfg.wan.shift, step=0.1, label="Flow shift")
                    fluidity = gr.Dropdown(
                        [f"{value}x - {cfg.wan.fps * value} fps" for value in FLUIDITY_CHOICES],
                        value=f"{cfg.wan.fluidity}x - {cfg.wan.fps * cfg.wan.fluidity} fps",
                        label="Fluidity (RIFE)",
                        info="Extra frames are estimated by optical flow between the sampled frames.",
                    )
                    quality = gr.Slider(1, 10, value=cfg.wan.quality, step=1, label="Video quality")
                    negative = gr.Textbox(value=WAN_NEGATIVE, label="Negative prompt", lines=2)
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        randomize = gr.Checkbox(value=True, label="Randomize")

                run_btn = gr.Button("Generate video", variant="primary")
                gr.Markdown(
                    f"fps {cfg.wan.fps}, frames {cfg.wan.min_frames}-{cfg.wan.max_frames}."
                    f" Duration maps onto that range."
                )

            with gr.Column(scale=4):
                output = gr.Video(label="Result", autoplay=True, elem_id="generated-video")
                status_line = gr.Markdown("")
                session = gr.Files(label="Session")

        def _fluidity_value(choice: str) -> int:
            try:
                return int(str(choice).split("x", 1)[0])
            except (TypeError, ValueError):
                return cfg.wan.fluidity

        def _rewrite(prompt_text, first_path):
            if not (prompt_text or "").strip():
                return prompt_text, "nothing to rewrite"
            images = []
            if isinstance(first_path, str):
                try:
                    images.append(prompting.shrink(first_path))
                except Exception:  # noqa: BLE001 - a missing frame just means no image
                    pass
            result = prompting.rewrite(prompt_text, images)
            if result.ok:
                return result.text, f"rewritten by {result.detail}"
            return result.text, f"unchanged - the call failed: `{result.detail}`"

        def _segment(first_path, hint):
            images = []
            if isinstance(first_path, str):
                try:
                    images.append(prompting.shrink(first_path))
                except Exception:  # noqa: BLE001
                    pass
            result = prompting.next_segment(images, hint=(hint or "").strip())
            if result.ok:
                return result.text, f"written by {result.detail}"
            return result.text, f"template used - the call failed: `{result.detail}`"

        def _grab(video_path, ts):
            return extract_frame(video_path, ts)

        def _use_last(video_path):
            frame = last_frame_of(video_path)
            if frame is None:
                raise gr.Error("no clip to take a frame from")
            return frame

        preset.change(lambda label: dict(PRESETS).get(label, ""), preset, prompt)
        rewrite_btn.click(_rewrite, [prompt, first], [prompt, prompt_status])
        segment_btn.click(_segment, [first, prompt], [prompt, prompt_status])
        grab_btn.click(fn=None, inputs=None, outputs=[timestamp], js=GRAB_TIMESTAMP_JS)
        timestamp.change(_grab, [output, timestamp], [first])
        last_btn.click(_use_last, [output], [first])

        # Wired explicitly so the fluidity label ("2x - 32 fps") is parsed once, at the edge.
        run_btn.click(
            lambda f, l, p, d, s, g, g2, n, sd, rnd, sch, fs, fl, q, progress=gr.Progress(): _render(
                engine, store, cfg, f, l, p, d, s, g, g2, n, sd, rnd, sch, fs,
                _fluidity_value(fl), q, progress=progress,
            ),
            [first, last, prompt, duration, steps, guidance, guidance_2, negative, seed,
             randomize, scheduler, flow_shift, fluidity, quality],
            [output, session, status_line],
        )
    return demo
