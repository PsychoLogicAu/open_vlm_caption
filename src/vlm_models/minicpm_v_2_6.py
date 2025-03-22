from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from vlm_models.base_model import BaseVLMModel


class MiniCPM_V_2_6(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "minicpm-v-2_6": "openbmb/MiniCPM-V-2_6",
            "minicpm-o-2_6": "openbmb/MiniCPM-o-2_6",
            "minicpm-llama3-v-2_5": "openbmb/MiniCPM-Llama3-V-2_5",
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
                        "vpm",
                        "resampler",
                    ]
                    if self.checkpoint == "openbmb/MiniCPM-V-2_6"
                    else (
                        [
                            "apm",
                            "resampler",
                            "tts",
                            "vpm",
                        ]
                        if self.checkpoint == "openbmb/MiniCPM-o-2_6"
                        else []
                    )
                ),
            )
            if self.quantize
            else None
        )
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )

    def _process_query(self, system_prompt, prompt):
        return f"{system_prompt}\n{prompt}"

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        msgs = [{"role": "user", "content": [image, self.query]}]
        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
        )
