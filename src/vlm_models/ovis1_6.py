from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import OvisGemma2GPTQForCausalLM
from PIL import Image
from transformers import GenerationConfig
import torch

from vlm_models.base_model import BaseVLMModel


class Ovis1_6Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        # TODO: support non-quantized models, not enough VRAM to test these
        checkpoint_mapping = {
            "ovis1.6": "AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4",  # Default checkpoint
            "ovis1.6-gemma2-9b": "AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4",
            "ovis1.6-llama3.2-3b": "AIDC-AI/Ovis1.6-Llama3.2-3B-GPTQ-Int4",
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
        self.model = OvisGemma2GPTQForCausalLM.from_quantized(
            self.checkpoint,
            torch_dtype=torch.float16,
            multimodal_max_length=8192,
            trust_remote_code=True,
        )
        self.model.model.generation_config = GenerationConfig.from_pretrained(
            self.checkpoint
        )

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return f"<image>\n{query}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            self.query, [image]
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [
            pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        ]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output
