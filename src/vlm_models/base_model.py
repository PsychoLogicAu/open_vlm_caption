from abc import ABC, abstractmethod
from PIL import Image
import imghdr
import logging
import math


class BaseVLMModel(ABC):
    def __init__(self, checkpoint: str, system_prompt: str, prompt: str, quantize: bool):
        self.checkpoint = checkpoint
        self.query = self._process_query(system_prompt, prompt)
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.supports_batch = False
        # TODO: downscale params from arguments / config
        self.image_downscale_max_dim=1280
        self.image_downscale_target_mp=1.0
        self.image_downscale_stride=64
        self._initialize_model()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug(str(self.model))

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        pass

    @abstractmethod
    def _process_query(self, system_prompt, prompt):
        """Process the query as required by the model."""
        pass

    @abstractmethod
    def _preprocess_image(self, img_path):
        """Load and process the image as required by the model."""
        pass

    @abstractmethod
    def _generate_response(self, image):
        """Generate a response given an image and a query."""
        pass

    def checkpoint_name(self):
        """Returns the name of the checkpoint."""
        return self.checkpoint

    def caption_image(self, img_path):
        """Generates a caption for an image using the model, and returns the response."""
        if imghdr.what(img_path) is None:
            raise ValueError(f"Invalid image: {img_path}")

        image = self._preprocess_image(img_path)
        return self._generate_response(image)

    def batch_caption_images(self, img_paths):
        if self.supports_batch:
            return self._generate_batch_response(img_paths)
        else:
            raise ValueError("Batch processing not supported by this model")
        
    def downscale_image(self, image):
        """
        Opens, converts, and resizes an image while maintaining aspect ratio.
        
        The logic prioritizes:
        1. Scaling down to meet the max_dim constraint.
        2. Scaling down (if needed) to meet the target_mp constraint.
        3. Ensuring dimensions are multiples of the stride (rounding down).
        4. Never upscaling the image beyond its original size.

        Args:
            img_path (str): The file path to the image.
            max_dim (int): The maximum allowed side length.
            target_mp (float): The desired image size in Megapixels (e.g., 1.0 for 1MP).
            stride (int): The new dimensions must be a multiple of this value.

        Returns:
            PIL.Image: The downscaled RGB image.
        """
        width, height = image.size
        
        # Target pixel count (e.g., 1.0 MP = 1,000,000 pixels)
        target_pixels = self.image_downscale_target_mp * 1_000_000.0
        current_pixels = width * height
        
        # 1. Determine the maximum allowed scale factor for downscaling (should be <= 1.0)
        
        # a. Constraint from max_dim
        scale_by_max_dim = min(self.image_downscale_max_dim / width, self.image_downscale_max_dim / height) if max(width, height) > self.image_downscale_max_dim else 1.0
        
        # b. Constraint from target_mp (using square root of ratio to find side scaling factor)
        # The scale factor for sides is the square root of the scale factor for area (pixels)
        scale_by_target_mp = 1.0
        if current_pixels > target_pixels:
            scale_by_target_mp = math.sqrt(target_pixels / current_pixels)
            
        # c. Choose the *smallest* scale factor to satisfy all constraints (and ensure downscaling)
        # This factor will be <= 1.0
        scale_factor = min(scale_by_max_dim, scale_by_target_mp)
        
        # 2. Compute the new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 3. Apply the stride (modulo) constraint
        
        def round_down_to_multiple(dim, stride):
            # Round the dimension down to the nearest multiple of stride
            return math.floor(dim / stride) * stride

        final_width = round_down_to_multiple(new_width, self.image_downscale_stride)
        final_height = round_down_to_multiple(new_height, self.image_downscale_stride)

        # Ensure final dimensions are at least 'stride' (prevents zero dimensions for tiny inputs)
        final_width = max(self.image_downscale_stride, final_width)
        final_height = max(self.image_downscale_stride, final_height)

        # 4. Resize and return
        # Use LANCZOS (high-quality) for downsampling
        resized_image = image.resize(
            (final_width, final_height), 
            Image.Resampling.LANCZOS
        )

        if scale_factor < 1.0:
            logging.debug(f"Downscaled image from ({width}, {height}) to ({final_width}, {final_height}).")
        
        return resized_image