import torch
from PIL import Image
import numpy as np
import folder_paths
import os
import sys

class HybridTaggingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tagging_mode": (["wd_tagger", "qwen_vl", "both", "hybrid"], {"default": "hybrid"}),
                "system_prompt": ("STRING", {
                    "default": "You are an expert image captioning AI that creates natural-language descriptions from booru-style tags and the image.",
                    "multiline": True
                }),
                "wd_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd_character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd_replace_underscore": ("BOOLEAN", {"default": True}),
                "wd_exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "qwen_model": ("STRING", {
                    "default": "Qwen/Qwen3-VL-4B-Instruct",
                    "multiline": False
                }),
                "qwen_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "qwen_max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "qwen_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("wd_tags", "qwen_caption", "combined_output")
    FUNCTION = "process_image"
    CATEGORY = "image/tagging"
    
    def __init__(self):
        self.qwen_model = None
        self.qwen_processor = None
        self.qwen_loaded = False
        
    def process_image(self, image, tagging_mode, system_prompt, wd_threshold, 
                     wd_character_threshold, wd_replace_underscore, wd_exclude_tags,
                     qwen_model, qwen_temperature, qwen_max_new_tokens, qwen_seed, 
                     device, keep_model_loaded):
        
        print(f"Processing image with mode: {tagging_mode}")
        
        # Convert image
        if len(image.shape) == 4:
            image_tensor = image
            image = image[0]
        else:
            image_tensor = image.unsqueeze(0)
            
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        wd_tags = ""
        qwen_caption = ""
        combined_output = ""
        
        # WD Tagger processing - placeholder for now
        if tagging_mode in ["wd_tagger", "both", "hybrid"]:
            wd_tags = self.run_wd_tagger(pil_image, wd_threshold, wd_character_threshold,
                                         wd_replace_underscore, wd_exclude_tags)
            print(f"WD tags: {wd_tags[:200]}")
        
        # Qwen VL processing
        if tagging_mode in ["qwen_vl", "both", "hybrid"]:
            print("Running Qwen3-VL...")
            if tagging_mode == "hybrid" and wd_tags:
                qwen_prompt = f"Create a natural-language caption based on these tags: {wd_tags}. Describe the image in detail."
            else:
                qwen_prompt = "Describe this image in detail."
                
            qwen_caption = self.run_qwen_vl(pil_image, qwen_prompt, system_prompt, 
                                          qwen_model, qwen_temperature, 
                                          qwen_max_new_tokens, qwen_seed, device)
            print(f"Qwen caption: {qwen_caption[:100]}...")
        
        # Create combined output
        if tagging_mode == "hybrid":
            combined_output = qwen_caption if qwen_caption and "NO_CAPTION" not in qwen_caption else wd_tags
        elif tagging_mode == "wd_tagger":
            combined_output = wd_tags
        elif tagging_mode == "qwen_vl":
            combined_output = qwen_caption
        else:  # both
            parts = []
            if wd_tags:
                parts.append(wd_tags)
            if qwen_caption and qwen_caption != "NO_CAPTION_GENERATED":
                parts.append(qwen_caption)
            combined_output = ". ".join(parts) if parts else "NO_OUTPUT_GENERATED"
        
        return (wd_tags, qwen_caption, combined_output)
    
    def run_wd_tagger(self, image, threshold, character_threshold, replace_underscore, exclude_tags):
        """Run a WD14 ONNX tagger from the comfyui-wd14-tagger models folder"""
        try:
            import csv
            import numpy as np
            import onnxruntime as ort
            from PIL import Image as PILImage

            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "comfyui-wd14-tagger", "models")
            preferred = ["wd-swinv2-tagger-v3", "wd-eva02-large-tagger-v3"]
            available = [f[:-5] for f in os.listdir(models_dir)
                         if f.endswith(".onnx") and os.path.exists(os.path.join(models_dir, f[:-5] + ".csv"))]
            name = next((m for m in preferred if m in available), available[0] if available else None)
            if name is None:
                raise FileNotFoundError(f"No WD14 .onnx + .csv pair found in {models_dir}")

            if getattr(self, "_wd_name", None) != name:
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if p in ort.get_available_providers()]
                self._wd_session = ort.InferenceSession(os.path.join(models_dir, name + ".onnx"), providers=providers)
                self._wd_name = name
            session = self._wd_session

            inp = session.get_inputs()[0]
            size = inp.shape[1]
            img = image.convert("RGB")
            ratio = float(size) / max(img.size)
            new_size = tuple(max(1, int(x * ratio)) for x in img.size)
            img = img.resize(new_size, PILImage.LANCZOS)
            square = PILImage.new("RGB", (size, size), (255, 255, 255))
            square.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
            arr = np.array(square).astype(np.float32)[:, :, ::-1]  # RGB -> BGR
            arr = np.expand_dims(arr, 0)

            tags, cats = [], []
            with open(os.path.join(models_dir, name + ".csv"), encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    tags.append(row[1])
                    cats.append(row[2])

            probs = session.run([session.get_outputs()[0].name], {inp.name: arr})[0][0]
            exclude = {t.strip().lower().replace("_", " ") for t in exclude_tags.split(",") if t.strip()}
            out = []
            for tag, cat, p in zip(tags, cats, probs):
                if (cat == "0" and p > threshold) or (cat == "4" and p > character_threshold):
                    t = tag.replace("_", " ") if replace_underscore else tag
                    if t.lower().replace("_", " ") not in exclude:
                        out.append((cat != "4", -float(p), t))
            out.sort()  # characters first, then by confidence
            return ", ".join(t for _, _, t in out)
        except Exception as e:
            print(f"WD tagger error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def run_qwen_vl(self, image, prompt, system_prompt, model_name, temperature, 
                   max_new_tokens, seed, device):
        """Run Qwen3-VL model"""
        try:
            # Check if model is already loaded
            if not self.qwen_loaded:
                print(f"Loading Qwen3-VL model from: {model_name}")
                self.load_qwen_model(model_name, device)
            
            if self.qwen_model is None:
                return "MODEL_LOAD_FAILED"
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Process with Qwen3-VL
            text = self.qwen_processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.qwen_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.qwen_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            if seed > 0:
                torch.manual_seed(seed)
                
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.qwen_processor.tokenizer.eos_token_id,
                    eos_token_id=self.qwen_processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.qwen_processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Qwen VL error: {e}")
            import traceback
            traceback.print_exc()
            return "NO_CAPTION_GENERATED"
    
    def load_qwen_model(self, model_path, device):
        """Load Qwen3-VL model"""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            import torch
            
            # Load the model
            self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            
            # Load the processor
            self.qwen_processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.qwen_loaded = True
            print("Qwen3-VL model loaded successfully!")
            
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please run: pip install --upgrade transformers qwen-vl-utils")
            self.qwen_model = None
            raise
        except Exception as e:
            print(f"Failed to load Qwen3-VL: {e}")
            self.qwen_model = None
            raise