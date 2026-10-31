"""Image BBox Layout (MOSS): upload + resize an image, draw boxes on it, give each box a prompt,
and get MOSS-Image's layout JSON as the prompt. With no boxes the prompt passes through unchanged."""
import os

import folder_paths

from .common import load_painted_mask, load_rgb_and_alpha, resize_pair, to_tensors
from .layout import MASK_MODES, box_rects, build_passes, build_prompt


class ImageBBoxLayout:
    CATEGORY = "image/layout"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK", "STRING")
    RETURN_NAMES = ("image", "painted", "prompt", "boxes", "passes")
    FUNCTION = "run"
    DESCRIPTION = (
        "Upload an image (resized, sides snapped to 16), then draw and prompt boxes and paint a mask in the editor. "
        "prompt is a MOSS-Image layout prompt, or just your text when there are no boxes. "
        "boxes = the plain boxes (for Invoke Inpainting's box_mask). painted = just what you painted, plus the mask "
        "input and any transparency in the image (for Invoke Inpainting's inpaint_mask). "
        "passes = one edit per box with its own prompt and strength (for Invoke Edit Boxes)."
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
                # Legacy: the node used to build a grown/feathered mask from these. Unused now, kept only so the
                # widget values saved in older workflows still line up. Hidden in the UI.
                "mask_mode": (list(MASK_MODES), {"default": "boxes + masks", "tooltip": "(unused)"}),
                "mask_grow": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "(unused)"}),
                "mask_feather": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "(unused)"}),
                # Managed by the editor; hidden there.
                "mask_file": ("STRING", {"default": "", "tooltip": "Set by the editor"}),
            },
            "optional": {"mask": ("MASK",)},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image or image == "none":
            return "No image loaded: click the upload box on the node."
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    @classmethod
    def IS_CHANGED(cls, image, mask_file="", **kwargs):
        try:
            stamp = [os.path.getmtime(folder_paths.get_annotated_filepath(image))]
            if mask_file:
                try:
                    stamp.append(os.path.getmtime(folder_paths.get_annotated_filepath(mask_file)))
                except Exception:
                    pass
            return tuple(stamp)
        except Exception:
            return float("NaN")

    def run(self, image, max_megapixels, prompt, bbox_data, mask_mode="boxes + masks",
            mask_grow=32, mask_feather=32, mask_file="", mask=None):
        rgb, alpha = load_rgb_and_alpha(folder_paths.get_annotated_filepath(image))
        rgb, alpha = resize_pair(rgb, alpha, max_megapixels, multiple=16, always_snap=True)
        img, alpha_mask = to_tensors(rgb, alpha)
        w, h = rgb.size
        import numpy as np
        import torch

        boxes = np.zeros((h, w), dtype="float32")
        for x0, y0, x1, y1 in box_rects(bbox_data, w, h):
            boxes[y0:y1, x0:x1] = 1.0
        ext = None
        if mask is not None:  # external mask: first item, resized to the image if needed
            ext_t = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[:1].unsqueeze(1).float()
            if ext_t.shape[-2:] != (h, w):
                ext_t = torch.nn.functional.interpolate(ext_t, size=(h, w), mode="bilinear")
            ext = ext_t[0, 0].cpu().numpy()
        if mask_file:  # mask painted with the node's own Paint tool
            try:
                pm = load_painted_mask(folder_paths.get_annotated_filepath(mask_file), (w, h))
            except Exception:
                pm = None
            if pm is not None:
                ext = pm if ext is None else np.maximum(ext, pm)
        boxes_t = torch.from_numpy(boxes).float().unsqueeze(0)  # hard boxes, no grow/feather
        painted = alpha_mask[0].cpu().numpy()  # transparency in the image
        if ext is not None:
            painted = np.maximum(painted, ext)
        painted_t = torch.from_numpy(np.ascontiguousarray(painted)).float().clamp(0, 1).unsqueeze(0)
        return (img, painted_t, build_prompt(bbox_data, w, h, prompt), boxes_t, build_passes(bbox_data, w, h, prompt))


NODE_CLASS_MAPPINGS = {"ImageBBoxLayout": ImageBBoxLayout}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBBoxLayout": "Image BBox Layout (MOSS)"}
