import os
import time
import random
import string
from datetime import datetime
from typing import Tuple, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Configuration Artifact
DEMO_MODES = {
    "Auto": {
        "prompt": "<image>\n<|grounding|>Extract all content from the document. Automatically detect the document type (text, table, chart, formula) and extract everything while preserving the original structure.",
        "desc": "Automatically detects document type and extracts all text, tables, and structures.",
        "base_size": 1024, "image_size": 640, "crop_mode": False
    }
}

class ImageTool:
    def __init__(self):
        self.model_id = "deepseek-ai/DeepSeek-OCR"
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, use_safetensors=True, attn_implementation="eager"
        ).to(dtype=torch.bfloat16, device="cuda").eval()

    def new_run_dir(self, base: str = "/content/runs") -> str:
        os.makedirs(base, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        rid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        path = os.path.join(base, f"run_{ts}_{rid}")
        os.makedirs(path)
        return path

    def run_ocr(self, image: Image.Image, mode: str = "Auto", custom_prompt: str = "", 
                base_size: int = 1024, image_size: int = 640, crop_mode: bool = False) -> Tuple[str, str, Optional[Image.Image]]:
        img = image.convert("RGB")
        if max(img.size) > 2000:
            s = 2000 / max(img.size)
            img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)
        run_dir = self.new_run_dir()
        img_path_proc = os.path.join(run_dir, "input.png")
        img.save(img_path_proc, optimize=True)
        if mode == "Custom Prompt" and custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = DEMO_MODES[mode]["prompt"]
        t0 = time.time()
        try:
            with torch.inference_mode():
                _ = self.model.infer(self.tok, prompt=prompt, image_file=img_path_proc, output_path=run_dir,
                                     base_size=base_size, image_size=image_size, crop_mode=crop_mode,
                                     save_results=True, test_compress=True)
        except ZeroDivisionError:
            print(" [Patched] Division by zero in compression ratio (valid_img_tokens==0). Ignored.")
        dt = time.time() - t0
        result_file = os.path.join(run_dir, "result.mmd")
        if not os.path.exists(result_file):
            result_file = os.path.join(run_dir, "result.txt")
        result = "[No text extracted]"
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                result = f.read().strip() or "[No text extracted]"
        boxed_path = os.path.join(run_dir, "result_with_boxes.jpg")
        boxed_img = Image.open(boxed_path) if os.path.exists(boxed_path) else None
        stats = f"**{dt:.1f}s** | Image: {img.size[0]}×{img.size[1]} px\n**Output directory:** {run_dir}\n"
        return result, stats, boxed_img

    def process(self, image_path: str) -> str:
        """Public method called by orchestrator. Returns only extracted text."""
        image = Image.open(image_path)
        text, stats, _ = self.run_ocr(image, mode="Auto")
        return text

if __name__ == "__main__":
    tool = ImageTool()
    # Example test
    # text, stats, _ = tool.run_ocr(Image.open("path_to_image.png"), mode="Auto")
    # print(text)
    pass