from PIL import Image
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    BitsAndBytesConfig,
)

from vlm_models.base_model import BaseVLMModel


class InstructBlipModel(BaseVLMModel):
    def __init__(self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None):
        checkpoint_mapping = {
            "instructblip": "Salesforce/instructblip-vicuna-7b", # Default checkpoint
            "instructblip-vicuna-7b": "Salesforce/instructblip-vicuna-7b",
            "instructblip-vicuna-13b": "Salesforce/instructblip-vicuna-13b",
            "instructblip-flan-t5-xl": "Salesforce/instructblip-flan-t5-xl",
            "instructblip-flan-t5-xxl": "Salesforce/instructblip-flan-t5-xxl",
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
                "query_tokens",
                "language_projection",
            ],
        ) if self.quantize else None
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.checkpoint,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = InstructBlipProcessor.from_pretrained(self.checkpoint)

    def _process_query(self, system_prompt, prompt):
        return f"{system_prompt}\n{prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        inputs = self.processor(images=image, text=self.query, return_tensors="pt").to(
            "cuda"
        )
        outputs = self.model.generate(
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
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
