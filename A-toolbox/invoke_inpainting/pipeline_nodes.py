"""Top-level pipeline nodes for the all-in-one Qwen edit workflow.

`QwenModeSelect`   choose Simple edit / Advanced edit / Model extraction + edit (and the advanced control type).
`QwenInputSelect`  image upload OR a saved character from the library (toggle) -> image (+ mesh/pose when a character is used).
`QwenLazyPick`     picks one of two inputs and only evaluates the chosen branch.
`QwenOutputSelect` lazy: only the subgraph for the chosen mode ever runs; returns image (+ mesh for the 3D preview).
`StepPreview`      pass-through image node that streams a thumbnail to the Live Preview node (js/live_preview.js)."""
import base64
import io
import os

import numpy as np
import torch

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker
from aiohttp import web
from server import PromptServer

from .character_lib import CharacterLoad, LIB_DIR, _lib_names

MODES = ["Simple edit", "Advanced edit", "Model extraction and edit"]


class _Any(str):
    def __ne__(self, o):
        return False


ANY = _Any("*")


class QwenModeSelect:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("mode", "mode_name")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (MODES, {"default": MODES[1], "tooltip": "Simple: Qwen edit from image + prompt only. Advanced: VLM-planned edit with segmentation and depth control. Model extraction: build (or load) a textured 3D character, then edit with it. Only the chosen subgraph is queued; the others are muted automatically."}),
        }}

    def run(self, mode):
        return (MODES.index(mode), mode)


class QwenInputSelect:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MESH", "MHR_POSE_DATA", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "mesh", "character", "use_library", "name")

    @classmethod
    def INPUT_TYPES(cls):
        d = folder_paths.get_input_directory()
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        names = _lib_names()
        return {"required": {
            "image": (sorted(files), {"image_upload": True}),
            "use_library": ("BOOLEAN", {"default": False, "label_on": "saved model", "label_off": "image", "tooltip": "off: use the uploaded image. on: use the saved character below (its source image replaces the input image; the mesh is reused, no extraction)"}),
            "character": (names if names else ["(library is empty)"], {"tooltip": "saved characters (press R to refresh the list)"}),
        }}

    @classmethod
    def IS_CHANGED(cls, image, use_library, character):
        if use_library:
            p = os.path.join(LIB_DIR, character, "mesh.pt")
            return f"lib:{character}:{os.path.getmtime(p) if os.path.isfile(p) else 0}"
        p = folder_paths.get_annotated_filepath(image)
        return f"img:{image}:{os.path.getmtime(p) if os.path.isfile(p) else 0}"

    def run(self, image, use_library, character):
        if use_library:
            mesh, pose, _thumb, src, name = CharacterLoad().run(character)
            return (src[..., :3], mesh, pose, True, name)
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(folder_paths.get_annotated_filepath(image))).convert("RGB")
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
        return (a, None, None, False, os.path.splitext(os.path.basename(image))[0])


POSE_DIR = os.path.join(folder_paths.models_dir, "poses")


def _pose_names():
    if not os.path.isdir(POSE_DIR):
        return []
    return sorted(d for d in os.listdir(POSE_DIR) if os.path.isfile(os.path.join(POSE_DIR, d, "pose.pt")))


class QwenPoseSelect:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "MHR_POSE_DATA", "BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("pose_image", "use_pose", "saved_pose", "use_saved", "name", "skip_save")

    @classmethod
    def INPUT_TYPES(cls):
        d = folder_paths.get_input_directory()
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        names = _pose_names()
        return {"required": {
            "image": (sorted(files), {"image_upload": True}),
            "use_pose": ("BOOLEAN", {"default": False, "label_on": "re-pose", "label_off": "keep A-pose", "tooltip": "Model extraction and edit only: re-pose the character (SAM 3D Body pose + the MHR rig)"}),
            "use_saved": ("BOOLEAN", {"default": False, "label_on": "saved pose", "label_off": "image", "tooltip": "take the pose from the pose library instead of the image"}),
            "pose": (names if names else ["(no saved poses)"], {"tooltip": "saved poses (press R / refresh to update)"}),
            "save_to_library": ("BOOLEAN", {"default": True, "tooltip": "when the pose comes from an image, save it to the pose library (named after the image)"}),
        }}

    @classmethod
    def IS_CHANGED(cls, image, use_pose, use_saved, pose, save_to_library):
        p = folder_paths.get_annotated_filepath(image)
        q = os.path.join(POSE_DIR, pose, "pose.pt")
        return f"{use_pose}:{use_saved}:{pose}:{image}:{os.path.getmtime(p) if os.path.isfile(p) else 0}:{os.path.getmtime(q) if os.path.isfile(q) else 0}"

    def run(self, image, use_pose, use_saved, pose, save_to_library):
        from PIL import Image, ImageOps
        blank = torch.zeros(1, 64, 64, 3)
        if not use_pose:
            return (blank, False, None, False, "", True)
        if use_saved:
            f = os.path.join(POSE_DIR, pose, "pose.pt")
            if not os.path.isfile(f):
                raise ValueError(f"saved pose '{pose}' not found in {POSE_DIR}")
            return (blank, True, torch.load(f, map_location="cpu", weights_only=False), True, pose, True)
        im = ImageOps.exif_transpose(Image.open(folder_paths.get_annotated_filepath(image))).convert("RGB")
        t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
        return (t, True, None, False, os.path.splitext(os.path.basename(image))[0], not save_to_library)


class PoseSave:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": "pose"}), "skip": ("BOOLEAN", {"default": False})},
                "optional": {"pose": ("MHR_POSE_DATA", {"lazy": True}), "image": ("IMAGE", {"lazy": True})}}

    def check_lazy_status(self, name, skip, pose=None, image=None):
        if skip:
            return []
        return [k for k, v in (("pose", pose), ("image", image)) if v is None]

    def run(self, name, skip, pose=None, image=None):
        if skip or pose is None:
            return {}
        safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in name).strip() or "pose"
        d = os.path.join(POSE_DIR, safe)
        os.makedirs(d, exist_ok=True)
        torch.save(pose, os.path.join(d, "pose.pt"))
        if image is not None:
            import torch.nn.functional as F
            from PIL import Image
            t = image[:1, ..., :3].permute(0, 3, 1, 2)
            h = 512
            t = F.interpolate(t, size=(h, max(1, int(t.shape[3] * h / t.shape[2]))), mode="bilinear", align_corners=False)
            Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(d, "thumb.png"))
        return {}


class QwenLazyPick:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("out",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"use_b": ("BOOLEAN", {"default": False}),
                             "a": (ANY, {"lazy": True}), "b": (ANY, {"lazy": True})}}

    def check_lazy_status(self, use_b, a=None, b=None):
        if use_b and b is None:
            return ["b"]
        if not use_b and a is None:
            return ["a"]
        return []

    def run(self, use_b, a=None, b=None):
        return (b if use_b else a,)


class QwenOutputSelect:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MESH")
    RETURN_NAMES = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"simple": ("IMAGE",), "advanced": ("IMAGE",), "extracted": ("IMAGE",), "model": ("MESH",)}}

    def run(self, simple=None, advanced=None, extracted=None, model=None):
        img = next((x for x in (extracted, advanced, simple) if x is not None), None)
        if img is None:
            raise ValueError("Output: no subgraph produced an image (is the chosen mode's subgraph connected?)")
        return (img, model if model is not None else ExecutionBlocker(None))


def _send_step(image, label):
        try:
            from PIL import Image
            x = image[0, ..., :3].clamp(0, 1).cpu().numpy()
            im = Image.fromarray((x * 255).astype(np.uint8))
            if max(im.size) <= 64 and float(x.max()) == 0.0:   # a lazy gate skipped this step: say so instead of showing black
                from PIL import ImageDraw
                im = Image.new("RGB", (512, 288), (38, 38, 44))
                ImageDraw.Draw(im).text((24, 130), "skipped - the planner did not need this step", fill=(190, 190, 200))
            s = 512.0 / max(im.size)
            if s < 1:
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            PromptServer.instance.send_sync("midfr.step_preview", {"label": label, "image": base64.b64encode(buf.getvalue()).decode()})
        except Exception as e:
            print(f"[StepPreview] {e}")


class StepPreview:
    """in-line: passes the image on and streams a thumbnail to the Live Preview node (only runs when something downstream needs it)"""
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "label": ("STRING", {"default": "step"})}}

    def run(self, image, label):
        _send_step(image, label)
        return (image,)


class StepPreviewSink:
    """output node: streams a thumbnail to the Live Preview node (replaces PreviewImage inside a subgraph)"""
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "label": ("STRING", {"default": "step"})}}

    def run(self, image, label):
        _send_step(image, label)
        return {}


NODE_CLASS_MAPPINGS = {"PoseSave": PoseSave, "QwenPoseSelect": QwenPoseSelect, "QwenModeSelect": QwenModeSelect, "QwenInputSelect": QwenInputSelect, "QwenLazyPick": QwenLazyPick,
                       "QwenOutputSelect": QwenOutputSelect, "StepPreview": StepPreview, "StepPreviewSink": StepPreviewSink}
NODE_DISPLAY_NAME_MAPPINGS = {"PoseSave": "Pose library: Save", "QwenPoseSelect": "Pose reference (optional)", "QwenModeSelect": "Mode Selection", "QwenInputSelect": "Input: image or saved model",
                              "QwenLazyPick": "Lazy pick (a / b)", "QwenOutputSelect": "Output (image + model)",
                              "StepPreview": "Step preview (in-line)", "StepPreviewSink": "Step preview (sink)"}


# ---- library gallery routes (used by js/library_gallery.js)
@PromptServer.instance.routes.get("/midfr/characters")
async def _midfr_characters(request):
    return web.json_response(_lib_names())


@PromptServer.instance.routes.get("/midfr/character_thumb")
async def _midfr_character_thumb(request):
    name = request.rel_url.query.get("name", "")
    if name not in _lib_names():
        return web.Response(status=404)
    d = os.path.join(LIB_DIR, name)
    for f in ("thumb.png", "source.png"):
        if os.path.isfile(os.path.join(d, f)):
            return web.FileResponse(os.path.join(d, f), headers={"Cache-Control": "no-cache"})
    return web.Response(status=404)


@PromptServer.instance.routes.get("/midfr/poses")
async def _midfr_poses(request):
    return web.json_response(_pose_names())


@PromptServer.instance.routes.get("/midfr/pose_thumb")
async def _midfr_pose_thumb(request):
    name = request.rel_url.query.get("name", "")
    f = os.path.join(POSE_DIR, name, "thumb.png")
    if name not in _pose_names() or not os.path.isfile(f):
        return web.Response(status=404)
    return web.FileResponse(f, headers={"Cache-Control": "no-cache"})
