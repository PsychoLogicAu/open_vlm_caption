from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

from vlm_models.base_model import BaseVLMModel

class Qwen2_5VLModel(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "qwen2.5": f"Qwen/Qwen2.5-VL-32B-Instruct{'-AWQ' if quantize else ''}",  # Default checkpoint
            "qwen2.5-vl-32b-instruct": f"Qwen/Qwen2.5-VL-32B-Instruct{'-AWQ' if quantize else ''}",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(
            checkpoint, system_prompt, prompt, quantize
        )  # Initialize the base class
        print(f"using query: {self.query}")
        self.supports_batch = False

    def _initialize_model(self):
        device_map = "cuda" if self.quantize else "auto"
        torch_dtype = torch.float16 if self.quantize else "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch_dtype,
            device_map=device_map,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            use_fast=False,
        )

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": None,
                },
                {"type": "text", "text": query},
            ],
        }

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        message = self.query
        message["content"][0]["image"] = image

        text = self.processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info([message])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response[0]
