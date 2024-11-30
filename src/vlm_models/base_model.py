from abc import ABC, abstractmethod
import imghdr
import logging


class BaseVLMModel(ABC):
    def __init__(self, checkpoint: str, query: str, quantize: bool):
        self.checkpoint = checkpoint
        self.query = self._process_query(query)
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self._initialize_model()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug(str(self.model))

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        pass

    @abstractmethod
    def _process_query(self, query):
        """Process the query as required by the model."""
        pass

    @abstractmethod
    def _process_image(self, img_path):
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

        image = self._process_image(img_path)
        return self._generate_response(image)
