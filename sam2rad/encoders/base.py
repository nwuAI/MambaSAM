from abc import abstractmethod, ABC
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    Inteface for image encoders.
    """

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        pass


class ImageEncoderFactory(ABC):
    """
    Interface for image encoder factories
    """

    @abstractmethod
    def build(self, args) -> ImageEncoder:
        pass
