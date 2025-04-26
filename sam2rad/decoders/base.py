from abc import ABC, abstractmethod
import torch.nn as nn


class MaskDecoder(nn.Module):
    """
    Interface for mask decoders.
    """

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        pass


class MaskDecoderFactory(ABC):
    """
    Interface for mask decoder factories
    """

    @abstractmethod
    def build(self, args) -> MaskDecoder:
        pass


