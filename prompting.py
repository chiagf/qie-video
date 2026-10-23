"""Prompt help for video: a vision model writes the motion, or a template does.

Two optional tasks, both shaped for *video* rather than stills:

    rewrite(prompt, images)  -> the user's instruction, made specific about motion over time
    next_segment(images, hint) -> "Next Segment: <camera move>, <what changes>. <style>."

The plumbing (HF Inference Providers, template fallback, sanitising) mirrors qie-studio's
prompting.py so both apps behave the same; only the *script* differs. A stills director writes
"add rim light"; a video director has to say where the camera goes and what the subject does
across the clip, because Wan2.2 renders motion, not a moment.

No token, no credits, or a provider outage returns a deterministic template and reports it in
`PromptResult.source`, so the UI can say so instead of silently handing back the input.
"""

from __future__ import annotations

import base64
import io
import os
import re
import zlib
from dataclasses import dataclass

from config import CONFIG, Config

MAX_SIDE = 1024

# A video prompt has to survive the whole clip: name the camera path first, then one action the
# subject performs, then the pacing. Identity must hold so Wan2.2 does not morph the subject.
REWRITE_SYSTEM = (
    "You are a video director writing a motion prompt for Wan2.2, an image-to-video diffusion "
    "model. Rewrite the user's instruction so the model can animate it. Open with the camera "
    "movement (dolly in, pan left, crane up, orbit, static), then the single action the subject "
    "performs across the clip, then the pacing (slow, steady, quick). Keep every name, object, "
    "location and intent the user gave; keep the subject's identity, wardrobe and lighting "
    "stable so nothing morphs mid-clip. Do not invent a different subject. Return ONLY the "
    "rewritten prompt - one line, no lists, no headings, no JSON, no preamble, no quotes, "
    "max {words} words."
)

# For chaining: given the last frame of a finished clip (and its prompt), write what the next
# clip does. Pairs with the "Use current frame as input" grab button.
NEXT_SEGMENT_SYSTEM = (
    "You are a storyboard director continuing a video. You are shown the last frame(s) of a "
    "clip, and you write the next segment as ONE line of plain text beginning with "
    "'Next Segment:'. Keep the same character(s), wardrobe, location, props and lighting; "
    "advance one story beat and change the camera move. Name the camera movement first, then "
    "what changes. No lists, no headings, no JSON, no preamble, no quotes, max {words} words, "
    "end with the style."
)

PRESETS: list[tuple[str, str]] = [
    (
        "Push in",
        "Next Segment: slow dolly-in on the same subject, same wardrobe and light, the subject "
        "blinks and breathes, the background falls slightly out of focus. Cinematic, 35mm.",
    ),
    (
        "Pan",
        "Next Segment: the camera pans steadily to reveal what sits beside the subject, the "
        "subject turning their head to follow, same horizon and lighting. Cinematic, 35mm.",
    ),
    (
        "Crane up",
        "Next Segment: the camera rises up and back from the subject, opening the space above "
        "them, the subject looking up, light falling the same way. Cinematic, 35mm.",
    ),
    (
        "Same framing",
        "Next Segment: identical framing and lighting, only the subject moves - one clear "
        "action, the environment untouched. Cinematic, 35mm.",
    ),
    (
        "Orbit",
        "Next Segment: the camera arcs around the subject at a steady pace, the subject holding "
        "still, the background sliding past, light unchanged. Cinematic, 35mm.",
    ),
]

_TEMPLATES = (
    "the camera pushes in slowly on the same subject, {hint} the light stays as it is and the "
    "background keeps its depth. Cinematic, 35mm.",
    "the camera drifts to the side of the same subject, {hint} same location, same wardrobe, "
    "one new detail entering the frame. Cinematic, 35mm.",
    "the camera steps back for a wider view of the same moment, {hint} the mood and lighting "
    "carry over unchanged. Cinematic, 35mm.",
)

_FENCE = re.compile(r"`{1,3}[a-zA-Z]*|`{1,3}")
_META = re.compile(
    r"^(?:here(?:'s| is| are)[^:.]*[:.]\s*|sure[\s,!.]*|certainly[\s,!.]*|"
    r"(?:rewritten|rewrite|output|instruction|prompt)\s*:\s*|next segment\s*:\s*|"
    r"next scene\s*:\s*)",
    re.IGNORECASE,
)
_STRUCT = re.compile(
    r"^\s*(?:ir|subject|action|camera|lighting|style|negative|scene\s*\d*|segment\s*\d*)\s*:\s*",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")


@dataclass
class PromptResult:
    text: str
    source: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.source == "vlm"


def sanitize(text: str, prefix: str = "", max_words: int = 60) -> str:
    """Flatten a model reply into one prompt line: no fences, no IR keys, no meta, word cap."""
    body = _FENCE.sub(" ", text or "")
    lines = []
    for raw in body.splitlines():
        line = raw.strip().strip('"').strip()
        if not line or line in {"{", "}", "[", "]"}:
            continue
        line = _STRUCT.sub("", line)
        lines.append(line)
    body = " ".join(lines)
    body = body.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    while True:
        stripped = _META.sub("", body).strip()
        if stripped == body:
            break
        body = stripped
    body = _SPACE.sub(" ", body).strip(" ,;.")
    if prefix:
        if body.lower().startswith(prefix.lower()):
            body = body[len(prefix) :].strip(" :")
    words = body.split(" ")
    if len(words) > max_words:
        body = " ".join(words[:max_words]).rstrip(" ,;")
    if not body:
        return prefix.strip()
    return f"{prefix}{body}." if prefix else f"{body}."


def _mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/png"


def to_data_url(item) -> str:
    """Accept a path or raw bytes, return an OpenAI-style data URL."""
    data = bytes(item) if isinstance(item, (bytes, bytearray)) else open(item, "rb").read()
    return f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def shrink(item, max_side: int = MAX_SIDE) -> bytes:
    """Re-encode an upload to at most `max_side` on the long edge, to keep image tokens sane."""
    from PIL import Image

    image = Image.open(io.BytesIO(bytes(item)) if isinstance(item, (bytes, bytearray)) else item)
    image = image.convert("RGB")
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(size, Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client(cfg: Config, provider: str):
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("no HF_TOKEN: cannot call Inference Providers")
    return InferenceClient(model=cfg.vlm_model, provider=provider, token=token)


def _completion(client, cfg: Config, system: str, prompt: str, images: list) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for item in images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(item)}})
    reply = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        max_tokens=200,
        temperature=0.6,
    )
    return reply.choices[0].message.content or ""


def _providers(cfg: Config) -> list[str]:
    order = [cfg.vlm_provider or "auto"]
    if "auto" not in order:
        order.append("auto")
    return order


def _call(cfg: Config, system: str, prompt: str, images: list, client=None) -> tuple[str, str]:
    """Try each provider in turn. Returns (text, "vlm:<provider+model>"); raises if all fail."""
    errors = []
    for provider in _providers(cfg):
        try:
            active = client if client is not None else _client(cfg, provider)
            return _completion(active, cfg, system, prompt, images), f"vlm:{provider}/{cfg.vlm_model}"
        except Exception as error:  # noqa: BLE001 - one provider failing is the normal case
            errors.append(f"{provider}: {type(error).__name__}: {error}")
    raise RuntimeError(" | ".join(errors))


def _template(hint: str) -> str:
    body = hint.strip() or "the subject keeps doing what it was doing"
    pick = zlib.crc32(body.encode("utf-8")) % len(_TEMPLATES)
    return _TEMPLATES[pick].format(hint=body.rstrip(".") + ".")


def rewrite(prompt: str, images: list | None = None, cfg: Config | None = None, client=None) -> PromptResult:
    """Make the user's instruction more specific about motion, keeping their intent and words."""
    cfg = cfg or CONFIG
    original = (prompt or "").strip()
    if not original:
        return PromptResult("", "template", "empty prompt")
    if not cfg.vlm_enabled:
        return PromptResult(original, "template", "QIE_VLM=off")
    system = REWRITE_SYSTEM.format(words=cfg.vlm_max_words)
    try:
        text, detail = _call(cfg, system, f"Instruction: {original}\n\nRewritten:", list(images or []), client=client)
        return PromptResult(sanitize(text, "", cfg.vlm_max_words), "vlm", detail)
    except Exception as error:  # noqa: BLE001 - degrade to the user's own words, but say why
        return PromptResult(original, "template", str(error)[:300])


def next_segment(images: list, hint: str = "", cfg: Config | None = None, client=None) -> PromptResult:
    """One line describing the next clip, given the last frame(s) of the one before it."""
    cfg = cfg or CONFIG
    hint = (hint or "").strip()
    if not cfg.vlm_enabled:
        return PromptResult(
            sanitize(_template(hint), "Next Segment: ", cfg.vlm_max_words), "template", "QIE_VLM=off"
        )
    if not images:
        return PromptResult(
            sanitize(_template(hint), "Next Segment: ", cfg.vlm_max_words), "template", "no image"
        )
    prompt = NEXT_SEGMENT_SYSTEM.format(words=cfg.vlm_max_words)
    if hint:
        prompt += f"\n\nWhat should happen next, in the user's words: {hint}"
    try:
        text, detail = _call(cfg, prompt, "Describe the next segment.", images, client=client)
        return PromptResult(sanitize(text, "Next Segment: ", cfg.vlm_max_words), "vlm", detail)
    except Exception as error:  # noqa: BLE001
        return PromptResult(
            sanitize(_template(hint), "Next Segment: ", cfg.vlm_max_words), "template", str(error)[:300]
        )
