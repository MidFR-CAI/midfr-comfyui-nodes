from .batch_image_loader import BatchImageLoader
from .hybrid_tagging import HybridTaggingNode
from .text_saver import SameDirectoryTextSaver
from .image_batch_queue import ImageBatchQueue

NODE_CLASS_MAPPINGS = {
    "BatchImageLoader": BatchImageLoader,
    "HybridTaggingNode": HybridTaggingNode,
    "SameDirectoryTextSaver": SameDirectoryTextSaver,
    "ImageBatchQueue": ImageBatchQueue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageLoader": "Batch Image Loader",
    "HybridTaggingNode": "Hybrid Tagging Node",
    "SameDirectoryTextSaver": "Same Directory Text Saver",
    "ImageBatchQueue": "Image Batch Queue",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']