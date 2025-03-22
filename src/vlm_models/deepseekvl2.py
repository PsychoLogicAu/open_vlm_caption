from PIL import Image
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from vlm_models.base_model import BaseVLMModel

from deepseek_vl2.models import DeepseekVLV2Processor


class DeepSeekVL2Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "deepseek-vl2-tiny": "deepseek-ai/deepseek-vl2-tiny",
            "deepseek-vl2-small": "deepseek-ai/deepseek-vl2-small",
            "deepseek-vl2": "deepseek-ai/deepseek-vl2",
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
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(self.checkpoint)
        self.tokenizer = self.processor.tokenizer

        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "image_newline",
                    "projector",
                    "view_seperator",
                    "vision",
                ],
            )
            if self.quantize
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            quantization_config=quantization_config,
            ).to(torch.bfloat16).cuda().eval()

    def _process_query(self, system_prompt, prompt):
        self.system_prompt = system_prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        return conversation

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        prepare_inputs = self.processor(
            conversations=self.conversation,
            images=[image],
            force_batchify=True,
            system_prompt=self.system_prompt

        ).to(self.model.device)
        prepare_inputs = prepare_inputs.to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language.generate(
            input_ids = prepare_inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,  ##change this to align with the official usage
            do_sample=False,  ##change this to align with the official usage
            use_cache=True  ##change this to align with the official usage
        )

        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
