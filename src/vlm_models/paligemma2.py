from PIL import Image
import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)

from vlm_models.base_model import BaseVLMModel


class PaliGemma2Model(BaseVLMModel):
    def __init__(self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None):
        checkpoint_mapping = {
            "paligemma2": "google/paligemma2-10b-ft-docci-448",  # Default checkpoint
            # the 'docci' checkpoints are fine-tuned on the DOCCI dataset, best for generating captions
            "paligemma2-3b-ft-docci-448":"google/paligemma2-3b-ft-docci-448",
            "paligemma2-10b-ft-docci-448": "google/paligemma2-10b-ft-docci-448",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(checkpoint, system_prompt, prompt, quantize)  # Initialize the base class

    def _initialize_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                # load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                # # FIXME: this appears unsupported for the model type
                # llm_int8_skip_modules=[
                #     "multi_modal_projector",
                #     "vision_tower",
                # ],
            )
            if self.quantize
            else None
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(self.checkpoint)

    def _process_query(self, system_prompt, prompt):
        return "caption en <image>" # from example code

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        # TODO: model is pretrained on specific image size, does it need to be resized here?
        # Can we tile the image and post-process?
        return image

    def _generate_response(self, image):
        model_inputs = (
            self.processor(text=self.query, images=image, return_tensors="pt")
            .to(torch.float16)
            .to(self.model.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded
