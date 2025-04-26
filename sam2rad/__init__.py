import sys
import os

# fmt: off
sys.path.append(os.path.abspath('sam2rad/models'))
print(sys.path)
# fmt: on

from .blob.misc import (
    AverageMeter,
    DotDict,
    convert_to_semantic,
    overlay_contours,
    colorize_mask,
)
from .datasets import DATASETS, get_dataloaders
from .losses import CompositeLoss, dice_loss, focal_loss
from .models.sam2rad.build_model import build_model as build_sam2rad
from .models.samrad.build_model import build_model as build_samrad
from .logging import setup_logging


setup_logging(output="logs", name="sam2rad")
