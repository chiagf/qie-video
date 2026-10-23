"""qie-video configuration: app knobs plus the engine's `WanSettings`.

The engine owns every `QIE_WAN_*` decision; this module adds only what the *app* needs (where
to write clips, how much history to keep, whether to load at boot) and exposes one public dict
for `/video-config` to print.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from qie_engine.settings import WanSettings, load as load_wan

MODES = ("video",)
OFFLOAD_MODES = ("auto", "model_cpu", "sequential", "none")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _env(name, "on" if default else "off").lower() in {"1", "on", "true", "yes"}


@dataclass(frozen=True)
class Config:
    mode: str
    out_dir: str
    max_history: int
    preload: bool
    auth: str
    is_space: bool
    vlm_enabled: bool
    vlm_provider: str
    vlm_model: str
    vlm_timeout: float
    vlm_max_words: int
    wan: WanSettings

    def public(self) -> dict:
        out = asdict(self)
        if out["auth"]:
            out["auth"] = "enabled"
        out["wan"] = self.wan.public()
        return out


def load() -> Config:
    return Config(
        mode=_env("QIE_MODE", "video"),
        out_dir=_env("QIE_OUT_DIR", "out"),
        max_history=_int("QIE_MAX_HISTORY", 20),
        preload=_bool("QIE_PRELOAD", True),
        auth=_env("QIE_AUTH", ""),
        is_space=bool(os.environ.get("SPACE_ID")),
        # The prompt writer is optional by design: no token, no credits or a provider outage
        # must all degrade to a deterministic template rather than fail a render.
        vlm_enabled=_bool("QIE_VLM", True),
        vlm_provider=_env("QIE_VLM_PROVIDER", "auto"),
        vlm_model=_env("QIE_VLM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"),
        vlm_timeout=_float("QIE_VLM_TIMEOUT", 90.0),
        vlm_max_words=_int("QIE_VLM_MAX_WORDS", 60),
        wan=load_wan(),
    )


CONFIG = load()
