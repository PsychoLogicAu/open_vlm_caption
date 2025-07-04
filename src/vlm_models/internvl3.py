from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
import torchvision.transforms as T

from vlm_models.base_model import BaseVLMModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL3Model(BaseVLMModel):
    def __init__(
        self, system_prompt: str, prompt: str, quantize: bool, checkpoint: str = None
    ):
        checkpoint_mapping = {
            "internvl3": "OpenGVLab/InternVL3-8B",  # Default checkpoint
            "internvl3-8b": "OpenGVLab/InternVL3-8B",
            "internvl3-9b": "OpenGVLab/InternVL3-9B",
        }
        checkpoint = checkpoint_mapping.get(checkpoint, None)
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint} not found. Available checkpoints are: {list(checkpoint_mapping.keys())}"
            )
        super().__init__(
            checkpoint, system_prompt, prompt, quantize
        )  # Initialize the base class
        print(f"using query: {self.query}")
        self.supports_batch = True

    def _initialize_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=[
                    "mlp1.0",
                    "mlp1.1",
                    "mlp1.3",
                    "vision_model.embeddings",
                    "vision_model.encoder",
                ],
            )
            if self.quantize
            else None
        )
        device_map = "auto"
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )

    def _process_query(self, system_prompt, prompt):
        query = f"{system_prompt}\n{prompt}"
        return f"<image>\n{query}"

    def _preprocess_image(self, img_path):
        max_tiles = 12
        pixel_values = load_image(img_path, max_num=max_tiles).to(torch.bfloat16).cuda()
        return pixel_values

    def _generate_response(self, image):
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        response = self.model.chat(self.tokenizer, image, self.query, generation_config)
        return response

    def _generate_batch_response(self, img_paths):
        with torch.autocast("cuda"):
            max_tiles = 12
            pixel_values_list = [
                load_image(img_path, max_num=max_tiles) for img_path in img_paths
            ]
            num_patches_list = [
                pixel_values.size(0) for pixel_values in pixel_values_list
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            questions = [self.query] * len(num_patches_list)
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config,
            )
            return responses
