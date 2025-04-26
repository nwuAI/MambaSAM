from typing import Any, Callable, List, Optional

import torch

from .dice_loss import dice_loss  # noqa: F401
from .focal_loss import focal_loss  # noqa: F401
from .iou_loss import iou_loss  # noqa: F401


class CompositeLoss:
    """
    Combines multiple loss functions into a single loss function.

    This class allows for the aggregation of multiple loss functions.

    Attributes:
        losses (List[Callable[..., torch.Tensor]]): A list of loss functions to be aggregated.
    """

    def __init__(
        self,
        loss_fns: List[Callable[..., torch.Tensor]],
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initializes the CompositeLoss with a list of loss functions.

        Parameters:
            losses (List[Callable[..., torch.Tensor]]): The loss functions to aggregate.
        """
        self.loss_fns = loss_fns
        self.weights = weights

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Computes the combined loss from the aggregated loss functions.

        Parameters:
            *args: Positional arguments passed to each loss function.
            **kwargs: Keyword arguments passed to each loss function.

        Returns:
            torch.Tensor: The computed combined loss as a tensor.
        """
        combined_loss = []
        weights = (
            self.weights
            if self.weights is not None
            else [1] * len(self.loss_fns)
        )
        assert len(weights) == len(
            self.loss_fns
        ), "Number of weights must match number of loss functions."
        for loss_fn, weight in zip(self.loss_fns, weights):
            combined_loss.append(weight * loss_fn(*args, **kwargs))
        combined_loss = torch.stack(combined_loss)

        return combined_loss.mean(0)
