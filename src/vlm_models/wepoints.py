from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
import torch

from vlm_models.base_model import BaseVLMModel


class WePOINTSModel(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "wepoints": "WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat",
            "points-1-5-qwen-2-5-7b-chat": "WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(
            checkpoint, system_prompt, prompt, quantize
        )  # Initialize the base class

    def _initialize_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "vision_encoder",
                    "vision_projector",
                ],
                bnb_8bit_compute_dtype=torch.float16,
            )
            if self.quantize
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        self.image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(
            self.checkpoint,
        )

    def _process_query(self, system_prompt, prompt):
        return f"{system_prompt}\n{prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        max_dim = 1024, 1024
        image.thumbnail(max_dim, Image.Resampling.LANCZOS)
        temp_path = "/tmp/image.png"
        image.save(temp_path, "PNG")
        return temp_path

    def _generate_response(self, image):
        content = [dict(type="image", image=image), dict(type="text", text=self.query)]
        messages = [{"role": "user", "content": content}]
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 0.0,
            "num_beams": 1,
        }
        response = self.model.chat(
            messages, self.tokenizer, self.image_processor, generation_config
        )
        return response
