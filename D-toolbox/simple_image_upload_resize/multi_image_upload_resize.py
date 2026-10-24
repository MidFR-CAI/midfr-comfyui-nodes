import io
import os
import asyncio
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

NUM_SLOTS = 9
NONE = "none"
THUMB_SIZE = 192
_thumb_cache = {}  # (path, mtime, size) -> jpeg bytes


def _make_thumb(path):
    with Image.open(path) as img:
        img.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))  # fast JPEG downscale
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/multi_image_upload/thumb")
    async def _thumb(request):
        name = request.rel_url.query.get("filename", "")
        sub = request.rel_url.query.get("subfolder", "")
        base = os.path.normpath(folder_paths.get_input_directory())
        path = os.path.normpath(os.path.join(base, sub, name))
        if not path.startswith(base) or not os.path.isfile(path):
            return web.Response(status=404)
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
        data = _thumb_cache.get(key)
        if data is None:
            try:
                data = await asyncio.get_running_loop().run_in_executor(None, _make_thumb, path)
            except Exception:
                return web.Response(status=415)
            if len(_thumb_cache) > 256:
                _thumb_cache.clear()
            _thumb_cache[key] = data
        return web.Response(
            body=data, content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
except Exception as e:  # keep nodes usable even if the route can't register
    print(f"[MultiImageUploadResize] thumbnail route not registered: {e}")


def resize_to_megapixels(img, max_megapixels):
    """Same logic as Simple Image Upload + Resize: only downsizes, dims divisible by 32."""
    width, height = img.size
    target_pixels = max_megapixels * 1000000
    current_pixels = width * height
    if current_pixels > target_pixels:
        scale = (target_pixels / current_pixels) ** 0.5
        new_width = ((int(width * scale) + 15) // 32) * 32
        new_height = ((int(height * scale) + 15) // 32) * 32
        new_width = max(32, new_width)
        new_height = max(32, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


class MultiImageUploadResize:
    """Upload up to 9 images, resize each to max megapixels, output ONE bundle
    (IMAGE_SET) to wire into 'Image Set Splitter (9)'."""

    @classmethod
    def INPUT_TYPES(cls):
        # Image slots are plain strings; the custom JS UI (js/multi_image_upload.js)
        # hides them and fills them in with the drag-and-drop boxes.
        optional = {
            f"image_{i}": ("STRING", {"default": NONE})
            for i in range(1, NUM_SLOTS + 1)
        }
        return {
            "required": {
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE_SET",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, max_megapixels, **kwargs):
        input_dir = folder_paths.get_input_directory()
        slots = []
        for i in range(1, NUM_SLOTS + 1):
            name = kwargs.get(f"image_{i}", NONE)
            if not name or name == NONE:
                slots.append(None)
                continue
            path = os.path.normpath(os.path.join(input_dir, name))
            if not path.startswith(os.path.normpath(input_dir)) or not os.path.isfile(path):
                slots.append(None)
                continue
            img = ImageOps.exif_transpose(Image.open(path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = resize_to_megapixels(img, max_megapixels)
            arr = np.array(img).astype(np.float32) / 255.0
            slots.append(torch.from_numpy(arr)[None,])
        return (slots,)

    @classmethod
    def IS_CHANGED(cls, max_megapixels, **kwargs):
        input_dir = folder_paths.get_input_directory()
        parts = [str(max_megapixels)]
        for i in range(1, NUM_SLOTS + 1):
            name = kwargs.get(f"image_{i}", NONE)
            p = os.path.join(input_dir, name) if name and name != NONE else None
            parts.append(f"{name}:{os.path.getmtime(p) if p and os.path.exists(p) else 0}")
        return "|".join(parts)


class ImageSetSplitter9:
    """Splits an IMAGE_SET into 9 fixed IMAGE outputs. Empty slots are dummy
    outputs: they return None, which optional image inputs treat as 'no image'."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE_SET",)}}

    RETURN_TYPES = tuple(["IMAGE"] * NUM_SLOTS)
    RETURN_NAMES = tuple(f"image_{i}" for i in range(1, NUM_SLOTS + 1))
    FUNCTION = "split"
    CATEGORY = "image"

    def split(self, images):
        out = []
        for i in range(NUM_SLOTS):
            t = images[i] if i < len(images) else None
            out.append(t)
        return tuple(out)


NODE_CLASS_MAPPINGS = {
    "MultiImageUploadResize": MultiImageUploadResize,
    "ImageSetSplitter9": ImageSetSplitter9,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiImageUploadResize": "Multi Image Upload + Resize (9)",
    "ImageSetSplitter9": "Image Set Splitter (9)",
}
