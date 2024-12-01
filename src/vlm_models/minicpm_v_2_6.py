from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from vlm_models.base_model import BaseVLMModel


class MiniCPM_V_2_6(BaseVLMModel):
    def __init__(self, query: str, quantize: bool, checkpoint: str = None):
        checkpoint = checkpoint or "openbmb/MiniCPM-V-2_6"
        super().__init__(checkpoint, query, quantize)  # Initialize the base class

    def _initialize_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "vpm",
                    "resampler",
                ],
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

    def _process_query(self, query):
        return query

    def _process_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        msgs = [{"role": "user", "content": [image, self.query]}]
        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
        )
