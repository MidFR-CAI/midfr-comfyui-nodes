import folder_paths
import torch
import numpy as np
from PIL import Image
import comfy.utils
import os

class SimpleImageUploadResize:
    """
    A simple node that combines image upload and resize functionality.
    Drag and drop an image, it will resize to target megapixels with dimensions divisible by 32.
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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "image"

    def process_image(self, image, max_megapixels):
        # Load the image
        image_path = os.path.join(folder_paths.get_input_directory(), image)
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get original dimensions
        width, height = img.size
        
        # Calculate target dimensions
        target_pixels = max_megapixels * 1000000
        
        # Calculate current pixels
        current_pixels = width * height
        
        # Only resize if image is larger than target
        if current_pixels > target_pixels:
            # Calculate scale factor to reach target pixels
            scale = (target_pixels / current_pixels) ** 0.5
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Adjust to be divisible by 32
            new_width = ((new_width + 15) // 32) * 32
            new_height = ((new_height + 15) // 32) * 32
            
            # Ensure minimum size
            new_width = max(32, new_width)
            new_height = max(32, new_height)
            
            # Resize with Lanczos
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert PIL image to tensor - Fixed version
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (img_tensor,)

    @classmethod
    def IS_CHANGED(cls, image, max_megapixels):
        # Return a hash of the image file to detect changes
        image_path = os.path.join(folder_paths.get_input_directory(), image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return float("NaN")

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
