from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from gptqmodel import GPTQModel

import torch

from vlm_models.base_model import BaseVLMModel


class Ovis2Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "ovis2-8b": "AIDC-AI/Ovis2-8B",
            "ovis2-34b": "AIDC-AI/Ovis2-34B-GPTQ-Int4",
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
        is_gptq = "gptq" in self.checkpoint.lower()
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "visual_tokenizer",
                    "vte",
                ],
            )
            if self.quantize and not is_gptq
            else None
        )
        device = "cuda"
        if is_gptq:
            self.model = GPTQModel.load(
                self.checkpoint,
                device=device,
                trust_remote_code=True,
            )
            self.model.model.generation_config = GenerationConfig.from_pretrained(
                self.checkpoint
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                quantization_config=quantization_config,
                multimodal_max_length=32768,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def _process_query(self, system_prompt, prompt):
        return f"<image>\n{system_prompt}\n{prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        max_partition = 9
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            self.query, [image], max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device="cuda")
        attention_mask = attention_mask.unsqueeze(0).to(device="cuda")
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device="cuda"
            )
        pixel_values = [pixel_values]

        # generate output
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
