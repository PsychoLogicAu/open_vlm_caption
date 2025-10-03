from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch

from vlm_models.base_model import BaseVLMModel

class YannQiRModel(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None, thinking: bool = False,
    ):
        checkpoint_mapping = {
            "yannqi-r": "YannQi/R-4B",  # Default checkpoint
            "yannqi-r-4b": "YannQi/R-4B",
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
        device_map = "cuda"
        torch_dtype = torch.float32
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
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

        chat = [message]

        text = self.processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode="auto" if self.thinking else "short"
        )

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=16384)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]

        # Decode output
        output_text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[1].strip()

        return output_text
