import copy
from PIL import Image
from openai import OpenAI
import base64
from io import BytesIO

from vlm_models.base_model import BaseVLMModel


class VLLM_OpenAI_Client(BaseVLMModel):
    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        checkpoint: str,
        openai_api_key: str,
        openai_api_base_url: str,
    ):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base_url,
        )
        super().__init__(
            checkpoint,
            system_prompt,
            prompt,
        )
        self.system_prompt = system_prompt
        self.user_prompt = prompt
        self.supports_batch = False

    def _initialize_model(self):
        pass

    def _encode_image_to_data_uri(self, image: Image.Image, format: str) -> str:
        """
        Encodes a PIL Image object into a Base64 Data URI string.
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        image_bytes = buffered.getvalue()

        base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        mime_type = f"image/{format.lower()}"
        data_uri = f"data:{mime_type};base64,{base64_encoded_image}"
        return data_uri

    def _process_query(self, system_prompt: str, prompt: str):
        pass

    def _preprocess_image(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        image = super().downscale_image(image)
        return image

    def _generate_response(self, image: Image.Image):
        data_uri = self._encode_image_to_data_uri(image, format="PNG")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.system_prompt}\n{self.user_prompt}",
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        chat_response = self.client.chat.completions.create(
            model=self.checkpoint,
            messages=messages,
        )

        print("Chat completion output:", chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content
