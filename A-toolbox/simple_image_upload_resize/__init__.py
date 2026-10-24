import folder_paths
import os

from .common import load_rgb_and_alpha, resize_pair, to_tensors


class SimpleImageUploadResize:
    """
    A simple node that combines image upload and resize functionality.
    Drag and drop an image, it will resize to target megapixels with dimensions divisible by 32.
    The MASK output comes from the image's alpha channel (a transparent PNG, or a mask painted
    with ComfyUI's mask editor: right-click the node -> Open in MaskEditor). No alpha -> empty mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                files.append(f)

        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "image"

    def process_image(self, image, max_megapixels):
        # get_annotated_filepath understands "name [input]" and sub-folders, which is how
        # the mask editor refers to the painted copy it saves.
        rgb, alpha = load_rgb_and_alpha(folder_paths.get_annotated_filepath(image))
        # Only downsizes; sides divisible by 32 when it does.
        rgb, alpha = resize_pair(rgb, alpha, max_megapixels, multiple=32, always_snap=False)
        return to_tensors(rgb, alpha)

    @classmethod
    def IS_CHANGED(cls, image, max_megapixels):
        # Re-run when the file changes on disk
        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(image))
        except Exception:
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        # Lets the mask editor's saved copy ("clipspace/... [input]") through, like Load Image.
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


# Register the node
NODE_CLASS_MAPPINGS = {
    "SimpleImageUploadResize": SimpleImageUploadResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleImageUploadResize": "Simple Image Upload + Resize"
}

# Multi-image upload + splitter nodes
from .multi_image_upload_resize import (
    NODE_CLASS_MAPPINGS as _MULTI_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MULTI_NAMES,
)
NODE_CLASS_MAPPINGS.update(_MULTI_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MULTI_NAMES)

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
