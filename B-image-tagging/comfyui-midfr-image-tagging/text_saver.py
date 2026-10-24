import os
import folder_paths

class SameDirectoryTextSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "", "multiline": False}),
                "wd_tags": ("STRING", {"default": "", "multiline": True}),
                "qwen_caption": ("STRING", {"default": "", "multiline": True}),
                "combined_output": ("STRING", {"default": "", "multiline": True}),
                "tagging_mode": (["wd_tagger", "qwen_vl", "both", "hybrid"], {"default": "hybrid"}),
                "source_directory": ("STRING", {"default": "", "multiline": False}),
                "use_wd_tags": ("BOOLEAN", {"default": True}),
                "use_qwen_caption": ("BOOLEAN", {"default": True}),
                "separator": ("STRING", {"default": ". ", "multiline": False}),
                "save_combined_only": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "save_text_file"
    CATEGORY = "image/tagging"
    OUTPUT_NODE = True
    
    def save_text_file(self, filename, wd_tags, qwen_caption, combined_output, tagging_mode, 
                       source_directory, use_wd_tags, use_qwen_caption, separator, save_combined_only):
        # Build the final text content based on mode
        if not filename:
            raise ValueError("Filename is required")
        
        # Determine what to save based on mode and options
        if save_combined_only:
            # Just save the combined output
            text_content = combined_output if combined_output else "NO_TAGS_GENERATED"
        else:
            # Save all outputs with labels
            parts = []
            if tagging_mode in ["wd_tagger", "both", "hybrid"] and use_wd_tags and wd_tags:
                parts.append(f"WD Tags: {wd_tags}")
            if tagging_mode in ["qwen_vl", "both", "hybrid"] and use_qwen_caption and qwen_caption:
                parts.append(f"Qwen Caption: {qwen_caption}")
            text_content = "\n\n".join(parts) if parts else "NO_TAGS_GENERATED"
        
        # Determine save path
        if source_directory and os.path.exists(source_directory):
            save_dir = source_directory
        else:
            # Try to find the source from the filename path
            save_dir = folder_paths.output_directory
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct the full save path
        save_path = os.path.join(save_dir, f"{filename}.txt")
        
        # Write the file
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Saved tags to: {save_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            raise
        
        return (save_path,)