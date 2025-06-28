from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch


from vlm_models.base_model import BaseVLMModel

def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
    return "", text

OUTPUT_FORMAT = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

class KimiVLModel(BaseVLMModel):
    def __init__(self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None):
        checkpoint_mapping = {
            "kimi-vl": "moonshotai/Kimi-VL-A3B-Thinking-2506", # Default checkpoint
            "kimi-vl-a3b-thinking-2506": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(checkpoint, system_prompt, prompt, quantize)  # Initialize the base class

    def _initialize_model(self):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_enable_fp32_cpu_offload=True,
        #     llm_int8_skip_modules=[
        #         "vision_model",
        #         "qformer",
        #         "language_projection",
        #     ],
        # ) if self.quantize else None
        quantization_config = None
        device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)

    def _process_query(self, system_prompt, prompt):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{system_prompt}\n{prompt}"}],
            },
        ]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # append the image into query
            query = self.query


            # Format the conversation
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
