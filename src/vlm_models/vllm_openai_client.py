import asyncio
import copy
from PIL import Image
from openai import OpenAI, AsyncOpenAI
import base64
from io import BytesIO

import logging
# Silence the httpx and openai loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
        # Synchronous client for single requests (if needed)
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base_url,
        )
        # Asynchronous client for batch requests
        self.async_client = AsyncOpenAI(  # <-- Initialize Async client
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
        self.supports_batch = True

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

    def _generate_batch_response(self, img_paths: list):
        """
        Generates responses for a list of image paths concurrently using asyncio.
        """
        
        async def generate_single_response_async(img_path: str):
            """Preprocesses image and sends a single asynchronous API call."""
            try:
                # 1. Preprocess the image (synchronous part)
                image = self._preprocess_image(img_path)
                data_uri = self._encode_image_to_data_uri(image, format="PNG")
                
                # 2. Prepare the message payload
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
                
                # 3. Send the API request asynchronously
                chat_response = await self.async_client.chat.completions.create(
                    model=self.checkpoint,
                    messages=messages,
                )
                
                return chat_response.choices[0].message.content
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                raise

        async def main():
            """Creates and runs all concurrent tasks."""
            tasks = [generate_single_response_async(path) for path in img_paths]
            # asyncio.gather runs all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run the asynchronous main function
        all_responses = asyncio.run(main())
        
        for i, response in enumerate(all_responses):
            print(f"Batch item {i} output: {response}")
            
        return all_responses
        
    async def caption_image_async(self, img_path: str):
        """Purely async call for a single image, no asyncio.run() inside."""
        # 1. Preprocess (Note: PIL open is sync, but fast enough)
        image = self._preprocess_image(img_path)
        data_uri = self._encode_image_to_data_uri(image, format="PNG")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{self.system_prompt}\n{self.user_prompt}"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }]
        
        # 2. Await the response from the async client
        chat_response = await self.async_client.chat.completions.create(
            model=self.checkpoint,
            messages=messages,
        )
        return chat_response.choices[0].message.content
