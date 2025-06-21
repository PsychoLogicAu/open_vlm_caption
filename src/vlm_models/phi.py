from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from vlm_models.base_model import BaseVLMModel


class PhiModel(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "phi-3-vision-128k-instruct": "microsoft/Phi-3-vision-128k-instruct",
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
                llm_int8_skip_modules=(
                    [
                        "model.embed_tokens",
                        "model.vision_embed_tokens",
                        "model.norm.weight",
                        "lm_head.weight",
                    ]
                    if self.checkpoint == "microsoft/Phi-3-vision-128k-instruct"
                    else []
                ),
            )
            if self.quantize
            else None
        )
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype="auto",
        ).cuda()

    def _process_query(self, system_prompt, prompt):
        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"
        return f"{user_prompt}<|image_1|>\n{system_prompt}\n{prompt}\n{prompt_suffix}{assistant_prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path)
        return image

    def _generate_response(self, image):
        inputs = self.processor(self.query, image, return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
