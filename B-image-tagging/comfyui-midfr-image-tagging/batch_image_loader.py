import os
import random
import glob
from PIL import Image
import numpy as np
import torch
import folder_paths

class BatchImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.png,*.jpg,*.jpeg,*.webp", "multiline": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "auto_increment": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "current_index", "total_images")
    FUNCTION = "load_images"
    CATEGORY = "image/tagging"
    
    # Class-level state: ComfyUI recreates node instances when IS_CHANGED forces a rerun,
    # so per-instance state would reset every run.
    _state = {}

    @classmethod
    def IS_CHANGED(cls, directory, pattern, shuffle, seed, image_index, auto_increment):
        if auto_increment:
            return float("nan")
        return (directory, pattern, shuffle, seed if shuffle else None, image_index)

    def load_images(self, directory, pattern, shuffle, seed, image_index, auto_increment):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        key = (os.path.abspath(directory), pattern, shuffle, seed if shuffle else None)
        st = self._state.get(key)
        if st is None:
            patterns = [p.strip() for p in pattern.split(",")]
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(directory, p)))
            files = sorted(set(files))
            if shuffle:
                random.Random(seed).shuffle(files)
            st = {"files": files, "next": None, "start": None}
            self._state[key] = st

        files = st["files"]
        if not files:
            self._state.pop(key, None)
            raise ValueError(f"No images found in {directory} with pattern {pattern}")

        # Start from image_index on first run, or whenever the user changes image_index
        if not auto_increment or st["next"] is None or st["start"] != image_index:
            current_index = image_index
            st["start"] = image_index
        else:
            current_index = st["next"]

        current_index %= len(files)
        st["next"] = (current_index + 1) % len(files)
        self.image_files = files
        print(f"[BatchImageLoader] image {current_index + 1}/{len(files)}: {os.path.basename(files[current_index])}")

        # Load the image at the current index
        img_path = files[current_index]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        return (image_tensor, filename, current_index, len(files))