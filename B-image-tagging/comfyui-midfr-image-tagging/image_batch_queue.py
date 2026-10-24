class ImageBatchQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_images": ("INT", {"default": 0, "min": 0}),
                "current_index": ("INT", {"default": 0, "min": 0}),
                "advance": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("next_index", "total_images_out", "is_complete")
    FUNCTION = "advance_queue"
    CATEGORY = "image/tagging"
    
    def __init__(self):
        self.last_index = -1
        self.completed = False
    
    def advance_queue(self, total_images, current_index, advance, reset):
        if reset:
            self.last_index = -1
            self.completed = False
            return (0, total_images, False)
        
        if total_images == 0:
            return (0, 0, True)
        
        # Determine next index
        if self.last_index == -1:
            # First run
            next_index = current_index
        elif advance:
            # Advance to next image
            next_index = (current_index + 1) % total_images
        else:
            # Keep current index
            next_index = current_index
        
        # Check if we've completed a full cycle
        is_complete = False
        if advance and next_index == 0 and current_index == total_images - 1:
            is_complete = True
            self.completed = True
        
        self.last_index = current_index
        
        return (next_index, total_images, is_complete)
    
    @classmethod
    def IS_CHANGED(cls, total_images, current_index, advance, reset):
        # Force node to update on each run
        import time
        return time.time()