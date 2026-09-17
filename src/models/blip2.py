import os
import time

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
)


class BLIP2Model:
    def __init__(
        self,
        base_path=None,
        adapter_path=None,
        load_in_4bit=True,
    ):
        self.base_path = base_path or os.getenv("SATQUERY_BLIP2_BASE")
        self.adapter_path = adapter_path or os.getenv("SATQUERY_BLIP2_ADAPTER")

        if not self.base_path or not self.adapter_path:
            raise ValueError("BLIP-2 base and adapter paths are required.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = Blip2Processor.from_pretrained(
            self.base_path,
            local_files_only=True,
        )

        quant_config = None

        if load_in_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.base_path,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True,
        )

        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path,
            local_files_only=True,
        ).eval()

    def _generate(self, image_path, prompt, max_new_tokens=80):
        start = time.perf_counter()

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        text = self.processor.batch_decode(
            output,
            skip_special_tokens=True,
        )[0].strip()

        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        return text, round((time.perf_counter() - start) * 1000, 2)

    def describe(self, image_path):
        return self._generate(
            image_path,
            "Describe the land cover and spatial patterns visible "
            "in this satellite image.",
        )

    def answer(self, image_path, question):
        return self._generate(
            image_path,
            question,
            max_new_tokens=50,
        )