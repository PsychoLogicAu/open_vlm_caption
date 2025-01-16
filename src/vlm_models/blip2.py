from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

from vlm_models.base_model import BaseVLMModel


class Blip2Model(BaseVLMModel):
    def __init__(self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None):
        checkpoint_mapping = {
            "blip2": "Salesforce/blip2-flan-t5-xl", # Default checkpoint
            "blip2-opt-6.7b-coco": "Salesforce/blip2-opt-6.7b-coco",
            "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
            "blip2-opt-6.7b": "Salesforce/blip2-opt-6.7b",
            "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",
            "blip2-flan-t5-xxl": "Salesforce/blip2-flan-t5-xxl",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(checkpoint, system_prompt, prompt, quantize)  # Initialize the base class

    def _initialize_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=[
                "vision_model",
                "qformer",
                "language_projection",
            ],
        ) if self.quantize else None
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = Blip2Processor.from_pretrained(self.checkpoint)

    def _process_query(self, system_prompt, prompt):
        return f"{system_prompt}\n{prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        inputs = self.processor(image, self.query, return_tensors="pt").to(
            "cuda"
        )
        out = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        return self.processor.decode(out[0], skip_special_tokens=True)[0].strip()
