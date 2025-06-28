from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    BitsAndBytesConfig,
)
import torch

from vlm_models.base_model import BaseVLMModel


class RevisualR1Model(BaseVLMModel):
    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        quantize: bool,
        checkpoint: str = None,
    ):
        checkpoint_mapping = {
            "revisual-r1": "csfufu/Revisual-R1-Coldstart",
            "revisual-r1-coldstart": "csfufu/Revisual-R1-Coldstart",
            "revisual-r1-final": "csfufu/Revisual-R1-final",
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
                    "visual",
                    "lm_head",
                ],
            )
            if self.quantize
            else None
        )
        self.dtype = torch.bfloat16
        '''
        MAX_TOKENS=16384
        TEMPERATURE=1.0
        TOP_P=0.95
        TOP_K=50
        NUM_RETURN_SEQUENCES=1


        prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        question="xxx"
        '''
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=self.dtype,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            dtype=self.dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def _process_query(self, system_prompt, user_prompt):
        model_specific_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n{user_prompt}\n{model_specific_prompt}"},
                ],
            }
        ]

    def _preprocess_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def _generate_response(self, image):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # Format the conversation
            convo_string = self.processor.apply_chat_template(
                self.query, tokenize=False, add_generation_prompt=True,
            )
            assert isinstance(convo_string, str)

            # Process the inputs
            inputs = self.processor(
                text=[convo_string],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            '''
            generated_ids = model.generate(**inputs,
                                           max_new_tokens=args.max_tokens,
                                           do_sample=args.do_sample,
                                           temperature=args.temperature,
                                           top_p=args.top_p,
                                           top_k=args.top_k,
                                           num_return_sequences=args.num_return_sequences)
            '''

            with torch.no_grad():
                # Generate the captions
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=16384,
                    do_sample=True,
                    # suppress_tokens=None,
                    # use_cache=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                )[0]

            # Trim off the prompt
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode the caption
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text.strip()

            return output_text
            