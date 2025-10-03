from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

from vlm_models.base_model import BaseVLMModel

def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
    return "", text

OUTPUT_FORMAT = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

class Kimi_VL_Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        # TODO: support non-quantized models, not enough VRAM to test these
        checkpoint_mapping = {
            "kimi-vl-a3b-thinking": "moonshotai/Kimi-VL-A3B-Thinking-2506",  # Default checkpoint
            "kimi-vl-a3b-thinking-2506": "moonshotai/Kimi-VL-A3B-Thinking-2506",
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return [
            # {
            #     "role": "system",
            #     "content": system_prompt,
            # },
            # {
            #     "role": "user",
            #     "content": prompt,
            # },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            },
        ]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        query = self.query
        query["content"].append({"type": "image", "image": image})

        text = self.processor.apply_chat_template(query, add_generation_prompt=True, return_tensors="pt",)
        inputs = self.processor(images=[image], text=text, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=32768, temperature=0.8)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
