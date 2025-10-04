from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import re
import torch

from vlm_models.base_model import BaseVLMModel


class Apriel_1_5_Model(BaseVLMModel):
    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        quantize: bool,
        checkpoint: str = None,
        thinking: bool = False,
    ):
        checkpoint_mapping = {
            "apriel-1.5": "ServiceNow-AI/Apriel-1.5-15b-Thinker",  # Default checkpoint
            "apriel-1.5-15b-thinker": "ServiceNow-AI/Apriel-1.5-15b-Thinker",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        self.thinking = thinking
        super().__init__(
            checkpoint, system_prompt, prompt, quantize
        )  # Initialize the base class
        print(f"using query: {self.query}")
        self.supports_batch = False

    def _initialize_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "multi_modal_projector",
                    "vision_tower",
                ],
            )
            if self.quantize
            else None
        )
                
        device_map = "auto"
        torch_dtype = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
        )

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        }]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        prompt = self.processor.apply_chat_template(
            self.query,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(
            self.model.device
        )
        inputs.pop("token_type_ids", None)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=16384, do_sample=True, temperature=0.6
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Raw output: {output}")
        if "BEGIN FINAL RESPONSE" in output:
            response = re.findall(
                r"\[BEGIN FINAL RESPONSE\](.*?)\[END FINAL RESPONSE\]", output, re.DOTALL
            )[0].strip()
        else:
            response = output

        return response
