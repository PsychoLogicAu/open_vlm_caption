from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
import torch

from vlm_models.base_model import BaseVLMModel


class JoyCaptionModel(BaseVLMModel):
    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        quantize: bool,
        checkpoint: str = None,
    ):
        checkpoint_mapping = {
            "joycaption": "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "joycaption-beta-one": "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "joycaption-alpha-two": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
            "llama-joycaption-alpha-two-hf-llava": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
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
                    "multi_modal_projector",
                    "vision_tower",
                ],
            )
            if self.quantize
            else None
        )
        self.dtype = torch.bfloat16
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=self.dtype,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            dtype=self.dtype,
        )

    def _process_query(self, system_prompt, prompt):
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = super().downscale_image(image)
        return image

    def _generate_response(self, image):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # Format the conversation
            # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
            # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
            # if not careful, which can make the model perform poorly.
            convo_string = self.processor.apply_chat_template(
                self.query, tokenize=False, add_generation_prompt=True
            )
            assert isinstance(convo_string, str)

            # Process the inputs
            inputs = self.processor(
                text=[convo_string], images=[image], return_tensors="pt"
            ).to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            with torch.no_grad():
                # Generate the captions
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    suppress_tokens=None,
                    use_cache=True,
                    temperature=0.6,
                    top_k=None,
                    top_p=0.9,
                )[0]

            # Trim off the prompt
            generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

            # Decode the caption
            caption = self.processor.tokenizer.decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            caption = caption.strip()
            return caption
