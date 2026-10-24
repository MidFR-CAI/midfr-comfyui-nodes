"""Image BBox Layout (MOSS): upload + resize an image, draw boxes on it, give each box a prompt,
and get MOSS-Image's layout JSON as the prompt. With no boxes the prompt passes through unchanged."""
import os

import folder_paths

from .common import load_rgb_and_alpha, resize_pair, to_tensors
from .layout import build_prompt


class ImageBBoxLayout:
    CATEGORY = "image/layout"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "prompt")
    FUNCTION = "run"
    DESCRIPTION = (
        "Upload an image (resized, sides snapped to 16), draw boxes on it and prompt each one. "
        "prompt is a MOSS-Image layout prompt, or just your text when there are no boxes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Managed by the node's UI (upload box); hidden there.
                "image": ("STRING", {"default": "", "tooltip": "Set by the upload box"}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "Description of the whole image"}),
                "bbox_data": ("STRING", {"default": "[]", "tooltip": "Boxes drawn in the editor (managed by the UI)"}),
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

    def run(self, image, max_megapixels, prompt, bbox_data):
        rgb, alpha = load_rgb_and_alpha(folder_paths.get_annotated_filepath(image))
        rgb, alpha = resize_pair(rgb, alpha, max_megapixels, multiple=16, always_snap=True)
        img, mask = to_tensors(rgb, alpha)
        w, h = rgb.size
        return (img, mask, build_prompt(bbox_data, w, h, prompt))


NODE_CLASS_MAPPINGS = {"ImageBBoxLayout": ImageBBoxLayout}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBBoxLayout": "Image BBox Layout (MOSS)"}
