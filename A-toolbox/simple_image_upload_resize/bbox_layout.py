"""Image BBox Layout (MOSS): upload + resize an image, draw boxes on it, give each box a prompt,
and get MOSS-Image's layout JSON out. No boxes -> the JSON output is blocked (nothing downstream runs)."""
import os

import folder_paths

from .common import load_rgb_and_alpha, resize_pair, to_tensors
from .layout import CANVASES, build_layout

try:  # present in current ComfyUI; lets us truly emit "no output"
    from comfy_execution.graph import ExecutionBlocker
except Exception:  # pragma: no cover
    ExecutionBlocker = None


class ImageBBoxLayout:
    CATEGORY = "image/layout"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "bbox_json")
    FUNCTION = "run"
    DESCRIPTION = (
        "Upload an image, resize it (sides snapped to 16), draw boxes on it and prompt each one. "
        "bbox_json is a MOSS-Image layout prompt; with no boxes it outputs nothing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Managed by the node's UI (upload button / drop zone); hidden there.
                "image": ("STRING", {"default": "", "tooltip": "Set by the upload box"}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "scene_prompt": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Whole-scene description and style (summary.caption)"}),
                "bbox_data": ("STRING", {"default": "[]",
                                         "tooltip": "Boxes drawn in the editor (managed by the UI)"}),
            },
            "optional": {
                "output_format": (["compact", "pretty"], {"default": "compact"}),
                "meta": ("STRING", {"default": "", "tooltip": "Optional comma-separated tags, e.g. poster"}),
                "canvas": (list(CANVASES), {"default": "single"}),
                "block_when_empty": ("BOOLEAN", {"default": True,
                                                 "tooltip": "No boxes: block bbox_json (nothing downstream runs). Off: output an empty string."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image or image == "none":
            return "No image loaded: click the upload box on the node."
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(image))
        except Exception:
            return float("NaN")

    def run(self, image, max_megapixels, scene_prompt, bbox_data, output_format="compact",
            meta="", canvas="single", block_when_empty=True):
        rgb, alpha = load_rgb_and_alpha(folder_paths.get_annotated_filepath(image))
        rgb, alpha = resize_pair(rgb, alpha, max_megapixels, multiple=16, always_snap=True)
        img, mask = to_tensors(rgb, alpha)
        w, h = rgb.size

        layout = build_layout(bbox_data, w, h, scene_prompt, meta, canvas, pretty=(output_format == "pretty"))
        if layout is None:
            layout = ExecutionBlocker(None) if (block_when_empty and ExecutionBlocker) else ""
        return (img, mask, w, h, layout)


NODE_CLASS_MAPPINGS = {"ImageBBoxLayout": ImageBBoxLayout}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBBoxLayout": "Image BBox Layout (MOSS)"}
