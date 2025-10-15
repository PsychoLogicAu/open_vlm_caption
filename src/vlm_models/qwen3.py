from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import torch

from vlm_models.base_model import BaseVLMModel


class Qwen3_VL_Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "qwen3-vl-4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-4b-thinking": "Qwen/Qwen3-VL-4B-Thinking",
            "qwen3-vl-8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3-vl-8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(
            checkpoint, system_prompt, prompt, quantize
        )  # Initialize the base class
        self.supports_batch = False

    def _initialize_model(self):
        torch_dtype = torch.bfloat16
        device_map = "cuda"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
        )

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        message = self.query
        message[0]["content"][0]["image"] = image

        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move all tensors in the inputs dictionary to the model's device
        device = self.model.device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Define Generation Hyperparameters
        generation_args = {
            "max_new_tokens": 4096,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 20,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            # "presence_penalty": 0.0,
        }
        generated_ids = self.model.generate(**inputs, **generation_args)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_text = response[0]
        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[1].strip()
        return output_text
