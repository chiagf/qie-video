"""Session history: every clip (and every grabbed frame) lands in a list you can feed back in.

Newest-first, capped at `max_history`, each entry keeping the prompt and seed that produced it.
Files live under `out_dir` so a rented box can mount a volume and keep them - the store never
assumes the disk is writable at boot, it falls back to a temp dir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    name: str
    path: str
    kind: str  # "clip" | "frame"
    prompt: str = ""
    seed: int = 0
    width: int = 0
    height: int = 0
    frames: int = 0
    fps: int = 0
    ms: int = 0
    ts: float = 0.0


class ClipStore:
    def __init__(self, out_dir: str, max_history: int = 20):
        self.root = Path(out_dir)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile

            self.root = Path(tempfile.mkdtemp(prefix="qie-video-out-"))
        self.max = max(1, int(max_history))
        self._items: list[Entry] = []
        self._counter = 0

    def job_dir(self) -> Path:
        self._counter += 1
        path = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}-{self._counter:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_clip(self, path: str, name: str = "clip.mp4", **meta) -> Entry:
        """Move a rendered mp4 under this store and list it."""
        return self._register(Path(path), name=name, kind="clip", **meta)

    def register_frame(self, image, name: str = "frame.png", **meta) -> Entry:
        """Save a frame extracted from a clip (the "use current frame as input" flow)."""
        path = self.job_dir() / name
        image.save(path, format="PNG")
        return self._register(path, name=name, kind="frame", **meta)

    def _register(self, source: Path, name: str, kind: str, **meta) -> Entry:
        if not source.is_file():
            raise ValueError(f"not a file: {source}")
        if kind == "clip":
            target = self.job_dir() / name
            source.replace(target)
            source = target
        entry = Entry(
            name=source.name,
            path=str(source),
            kind=kind,
            prompt=str(meta.get("prompt", ""))[:400],
            seed=int(meta.get("seed", 0)),
            width=int(meta.get("width", 0)),
            height=int(meta.get("height", 0)),
            frames=int(meta.get("frames", 0)),
            fps=int(meta.get("fps", 0)),
            ms=int(meta.get("ms", 0)),
            ts=time.time(),
        )
        self._items.insert(0, entry)
        # History is capped in memory AND on disk: the eviction used to drop the reference
        # only, so /data/out grew one clip per render forever.
        evicted = self._items[self.max :]
        del self._items[self.max :]
        for old in evicted:
            try:
                Path(old.path).unlink(missing_ok=True)
            except OSError:  # noqa: BLE001 - a clip we cannot delete is not worth failing for
                pass
        return entry

    def items(self) -> list[Entry]:
        return list(self._items)

    def clips(self) -> list[Entry]:
        return [entry for entry in self._items if entry.kind == "clip"]

    def paths(self) -> list[str]:
        return [entry.path for entry in self._items]

    def latest(self, kind: str | None = None) -> Entry | None:
        for entry in self._items:
            if kind is None or entry.kind == kind:
                return entry
        return None

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
