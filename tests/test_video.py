"""CPU-only tests for qie-video. No torch, no gradio, no network. Run from the repo root:

    python -m unittest discover -s tests -v

The engine itself is tested in the `vendor/qie-engine` submodule; these cover the app layer:
config, the video-specific prompt writer, the session store, and static checks on the modules
that need gradio (so a typo is caught here instead of as a 500 on the box).
"""

from __future__ import annotations

import ast
import builtins
import os
import pathlib
import py_compile
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "qie-engine"))

import config  # noqa: E402
import prompting  # noqa: E402
import store  # noqa: E402

MODULES = (
    "app.py",
    "config.py",
    "prompting.py",
    "store.py",
    "video_ui.py",
)

WAN_ENV = (
    "QIE_MODE",
    "QIE_OUT_DIR",
    "QIE_MAX_HISTORY",
    "QIE_PRELOAD",
    "QIE_AUTH",
    "QIE_VLM",
    "QIE_VLM_PROVIDER",
    "QIE_VLM_MODEL",
    "QIE_VLM_TIMEOUT",
    "QIE_VLM_MAX_WORDS",
    "QIE_WAN_STACK",
    "QIE_WAN_DTYPE",
    "QIE_WAN_OFFLOAD",
    "QIE_ATTN",
    "QIE_WAN_STEPS",
    "QIE_WAN_SHIFT",
    "QIE_WAN_FPS",
    "QIE_WAN_MIN_FRAMES",
    "QIE_WAN_MAX_FRAMES",
    "QIE_WAN_MAX_MP",
    "QIE_WAN_SCHEDULER",
    "QIE_WAN_FLUIDITY",
    "QIE_WAN_QUALITY",
    "QIE_WAN_RIFE_ENABLED",
)


def clean_env():
    for key in WAN_ENV:
        os.environ.pop(key, None)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Reply:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content: str, calls: list):
        self.content = content
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Reply(self.content)


class _Chat:
    def __init__(self, content: str, calls: list):
        self.completions = _Completions(content, calls)


class FakeClient:
    """Stands in for huggingface_hub.InferenceClient (only .chat.completions.create is used)."""

    def __init__(self, content: str = "Next Segment: slow dolly in, the subject turns. Cinematic."):
        self.calls: list = []
        self.chat = _Chat(content, self.calls)


class TestConfig(unittest.TestCase):
    def setUp(self):
        clean_env()

    def tearDown(self):
        clean_env()

    def test_defaults(self):
        cfg = config.load()
        self.assertEqual(cfg.mode, "video")
        self.assertEqual(cfg.out_dir, "out")
        self.assertEqual(cfg.max_history, 20)
        self.assertTrue(cfg.preload)
        self.assertEqual(cfg.auth, "")
        self.assertFalse(cfg.is_space)

    def test_engine_settings_are_inherited(self):
        cfg = config.load()
        self.assertEqual(cfg.wan.dtype, "fp8")
        self.assertEqual(cfg.wan.scheduler, "FlowMatchEulerDiscrete")
        self.assertEqual(cfg.wan.fluidity, 1)

    def test_vlm_defaults(self):
        cfg = config.load()
        self.assertTrue(cfg.vlm_enabled)
        self.assertEqual(cfg.vlm_provider, "auto")
        self.assertEqual(cfg.vlm_model, "Qwen/Qwen2.5-VL-72B-Instruct")
        self.assertEqual(cfg.vlm_max_words, 60)

    def test_vlm_env_overrides(self):
        os.environ["QIE_VLM"] = "off"
        os.environ["QIE_VLM_MODEL"] = "Qwen/Qwen2.5-VL-32B-Instruct"
        cfg = config.load()
        self.assertFalse(cfg.vlm_enabled)
        self.assertEqual(cfg.vlm_model, "Qwen/Qwen2.5-VL-32B-Instruct")

    def test_bad_int_falls_back(self):
        os.environ["QIE_MAX_HISTORY"] = "many"
        self.assertEqual(config.load().max_history, 20)

    def test_space_detection(self):
        os.environ["SPACE_ID"] = "gchia/qie-video"
        try:
            self.assertTrue(config.load().is_space)
        finally:
            os.environ.pop("SPACE_ID", None)

    def test_public_hides_auth_and_nests_wan(self):
        os.environ["QIE_AUTH"] = "me:secret"
        try:
            public = config.load().public()
        finally:
            os.environ.pop("QIE_AUTH", None)
        self.assertEqual(public["auth"], "enabled")
        self.assertNotIn("secret", str(public))
        self.assertIn("wan", public)
        self.assertEqual(public["wan"]["dtype"], "fp8")


class TestPromptSanitize(unittest.TestCase):
    def test_prefix_and_period(self):
        self.assertEqual(
            prompting.sanitize("dolly in on the cat", "Next Segment: ", 40),
            "Next Segment: dolly in on the cat.",
        )

    def test_existing_prefix_is_not_duplicated(self):
        self.assertEqual(
            prompting.sanitize("Next Segment: dolly in.", "Next Segment: ", 40),
            "Next Segment: dolly in.",
        )

    def test_both_prefixes_are_stripped(self):
        self.assertEqual(prompting.sanitize("Next Scene: pan left.", "", 40), "pan left.")

    def test_fences_and_meta_removed(self):
        raw = "```json\nHere is the rewritten prompt: a cat on a blue chair.\n```"
        self.assertEqual(prompting.sanitize(raw, "", 40), "a cat on a blue chair.")

    def test_struct_keys_stripped(self):
        self.assertEqual(
            prompting.sanitize("camera: pan left\nlighting: dusk\naction: turns head", "", 40),
            "pan left dusk turns head.",
        )

    def test_word_cap(self):
        text = " ".join(["word"] * 100)
        self.assertEqual(len(prompting.sanitize(text, "", 5).split()), 5)

    def test_empty_returns_prefix_only(self):
        self.assertEqual(prompting.sanitize("", "Next Segment: ", 40), "Next Segment:")


class TestPromptCalls(unittest.TestCase):
    def setUp(self):
        clean_env()
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACE_TOKEN", None)

    def tearDown(self):
        clean_env()

    def test_presets_exist_and_are_prefixed(self):
        self.assertGreaterEqual(len(prompting.PRESETS), 4)
        for label, text in prompting.PRESETS:
            self.assertTrue(label)
            self.assertTrue(text.startswith("Next Segment:"), label)

    def test_no_token_degrades_to_template_and_says_so(self):
        result = prompting.next_segment([b"\x89PNG\r\n\x1a\nfake"], hint="she picks up the cup")
        self.assertEqual(result.source, "template")
        self.assertFalse(result.ok)
        self.assertTrue(result.text.startswith("Next Segment: "))
        self.assertIn("picks up the cup", result.text)
        self.assertIn("HF_TOKEN", result.detail)

    def test_vision_call_used_when_client_is_given(self):
        client = FakeClient("Next Segment: crane up, the street empties. Cinematic, 35mm.")
        result = prompting.next_segment([b"\x89PNG\r\n\x1a\nfake"], client=client)
        self.assertEqual(result.source, "vlm")
        self.assertIn("vlm:", result.detail)
        self.assertTrue(result.text.startswith("Next Segment: crane up"))
        self.assertEqual(len(client.calls), 1)
        parts = client.calls[0]["messages"][1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_rewrite_returns_model_text(self):
        client = FakeClient("slow dolly in on a weathered fisherman mending a net, 50mm")
        result = prompting.rewrite("old man fixing net", client=client)
        self.assertEqual(result.source, "vlm")
        self.assertEqual(result.text, "slow dolly in on a weathered fisherman mending a net, 50mm.")

    def test_rewrite_without_token_keeps_user_text(self):
        result = prompting.rewrite("old man fixing net")
        self.assertEqual(result.source, "template")
        self.assertEqual(result.text, "old man fixing net")

    def test_rewrite_empty_prompt_is_a_noop(self):
        result = prompting.rewrite("   ")
        self.assertEqual(result.source, "template")
        self.assertEqual(result.text, "")

    def test_vlm_disabled_uses_template(self):
        os.environ["QIE_VLM"] = "off"
        cfg = config.load()
        result = prompting.next_segment([b"\x89PNG\r\n\x1a\nfake"], cfg=cfg)
        self.assertEqual(result.source, "template")
        self.assertIn("QIE_VLM", result.detail)

    def test_rewrite_system_asks_for_motion_not_a_still(self):
        text = prompting.REWRITE_SYSTEM.format(words=60).lower()
        self.assertIn("camera", text)
        self.assertIn("action", text)
        self.assertIn("pacing", text)
        self.assertIn("identity", text)

    def test_next_segment_system_keeps_identity(self):
        text = prompting.NEXT_SEGMENT_SYSTEM.format(words=60).lower()
        self.assertIn("next segment:", text)
        self.assertIn("keep the same", text)
        self.assertIn("story beat", text)

    def test_data_url_sniffs_jpeg(self):
        self.assertTrue(
            prompting.to_data_url(b"\xff\xd8\xff\xe0rest").startswith("data:image/jpeg;base64,")
        )


class TestClipStore(unittest.TestCase):
    def test_history_is_newest_first_and_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = store.ClipStore(tmp, max_history=2)
            source = pathlib.Path(tmp) / "render.mp4"
            source.write_bytes(b"mp4")
            first = box.register_clip(str(source), name="a.mp4", prompt="one", seed=1)
            source.write_bytes(b"mp4")
            second = box.register_clip(str(source), name="b.mp4", prompt="two", seed=2)
            source.write_bytes(b"mp4")
            third = box.register_clip(str(source), name="c.mp4", prompt="three", seed=3)
            self.assertEqual(len(box), 2)
            self.assertEqual([entry.seed for entry in box.items()], [3, 2])
            self.assertEqual(box.latest().path, third.path)
            self.assertNotIn(first.path, box.paths())
            self.assertTrue(pathlib.Path(second.path).exists())
            self.assertEqual(box.latest("clip").kind, "clip")

    def test_register_clip_moves_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "render.mp4"
            source.write_bytes(b"mp4")
            box = store.ClipStore(tmp, max_history=5)
            entry = box.register_clip(str(source), name="clip-7.mp4", prompt="pan left", seed=7)
            self.assertFalse(source.exists())
            self.assertTrue(pathlib.Path(entry.path).exists())
            self.assertEqual(entry.kind, "clip")
            self.assertEqual(entry.name, "clip-7.mp4")
            self.assertEqual(entry.prompt, "pan left")
            self.assertEqual(entry.seed, 7)

    def test_register_clip_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                store.ClipStore(tmp).register_clip(str(pathlib.Path(tmp) / "nope.mp4"))

    def test_register_frame_saves_a_png(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            box = store.ClipStore(tmp, max_history=5)
            entry = box.register_frame(Image.new("RGB", (16, 16), (0, 0, 0)), name="frame-3.png", seed=3)
            self.assertEqual(entry.kind, "frame")
            self.assertTrue(entry.path.endswith(".png"))
            self.assertTrue(pathlib.Path(entry.path).exists())

    def test_clips_filters_out_frames(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            box = store.ClipStore(tmp, max_history=5)
            box.register_frame(Image.new("RGB", (8, 8)), name="f.png")
            source = pathlib.Path(tmp) / "r.mp4"
            source.write_bytes(b"mp4")
            box.register_clip(str(source), name="c.mp4")
            self.assertEqual(len(box), 2)
            self.assertEqual(len(box.clips()), 1)
            self.assertEqual(box.clips()[0].kind, "clip")

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = store.ClipStore(tmp, max_history=5)
            source = pathlib.Path(tmp) / "r.mp4"
            source.write_bytes(b"mp4")
            box.register_clip(str(source))
            box.clear()
            self.assertEqual(len(box), 0)
            self.assertIsNone(box.latest())

    def test_job_dirs_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = store.ClipStore(tmp, max_history=5)
            self.assertNotEqual(box.job_dir(), box.job_dir())

    def test_unwritable_out_dir_falls_back_to_temp(self):
        box = store.ClipStore(os.path.join(os.sep, "proc", "definitely", "not", "writable"), 5)
        self.assertTrue(box.root.exists())


class TestVideoHelpers(unittest.TestCase):
    """The frame-grab helpers are pure cv2/PIL; check the degenerate paths without a video."""

    def test_extract_frame_is_none_without_a_path(self):
        import video_ui

        self.assertIsNone(video_ui.extract_frame(None, 0.0))
        self.assertIsNone(video_ui.extract_frame("", 1.0))

    def test_last_frame_is_none_without_a_path(self):
        import video_ui

        self.assertIsNone(video_ui.last_frame_of(None))

    def test_pick_seed_randomises(self):
        import video_ui

        self.assertGreaterEqual(video_ui._pick_seed(0, True), 0)
        self.assertEqual(video_ui._pick_seed(7, False), 7)
        # A negative seed means "randomise", so it must land inside the valid range.
        seed = video_ui._pick_seed(-1, False)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)

    def test_status_line_reports_interpolation_and_scheduler(self):
        import video_ui

        class Result:
            width, height, frames, fps, ms, peak_gb, seed = 624, 832, 162, 32, 9000, 30.5, 7
            interpolated = "rife"
            scheduler = "UniPCMultistep"

        line = video_ui._status_line(Result(), 12.0)
        self.assertIn("624 x 832", line)
        self.assertIn("162 frames @32 fps", line)
        self.assertIn("interpolated `rife`", line)
        self.assertIn("UniPCMultistep", line)


class TestNoUndefinedNames(unittest.TestCase):
    """app.py and video_ui.py need gradio, so the CPU suite cannot import them - but it can
    still catch a NameError, which would otherwise only show up as a request-time 500."""

    def _undefined(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        known = bound | set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__"}
        return sorted(name for name in used if name not in known)

    def test_no_undefined_names(self):
        for name in MODULES:
            with self.subTest(module=name):
                self.assertEqual(self._undefined(ROOT / name), [])


class TestTree(unittest.TestCase):
    def test_every_module_compiles(self):
        for name in MODULES:
            path = ROOT / name
            self.assertTrue(path.exists(), name)
            py_compile.compile(str(path), doraise=True)

    def test_container_ships_every_app_module(self):
        """app.py imports these, so a missing COPY line is a ModuleNotFoundError at runtime."""
        needed = {
            "app.py",
            "config.py",
            "prompting.py",
            "store.py",
            "video_ui.py",
            "README.md",
        }
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        copied = set()
        for line in text.splitlines():
            if line.startswith("COPY "):
                copied.update(token for token in line.split()[1:] if not token.startswith("./"))
        self.assertTrue(needed.issubset(copied), f"missing from Dockerfile COPY: {needed - copied}")

    def test_container_installs_the_engine_submodule(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY vendor/qie-engine", text)
        self.assertIn("pip install --no-cache-dir ./vendor/qie-engine", text)

    def test_workflow_checks_out_submodules(self):
        text = (ROOT / ".github" / "workflows" / "build-push.yml").read_text(encoding="utf-8")
        self.assertIn("submodules: true", text)

    def test_submodule_is_pinned(self):
        self.assertTrue((ROOT / ".gitmodules").exists())
        self.assertTrue((ROOT / "vendor" / "qie-engine" / "pyproject.toml").exists())

    def test_engine_is_not_vendored_directly(self):
        """The app must import the engine from the submodule, never a second local copy."""
        self.assertFalse((ROOT / "qie_engine").exists(), "a local qie_engine/ would shadow the submodule")

    def test_no_hardcoded_token(self):
        import re

        for name in MODULES:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{30,}", text), name)


if __name__ == "__main__":
    unittest.main()
