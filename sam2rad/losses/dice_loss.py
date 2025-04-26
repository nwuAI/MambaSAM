import torch


def dice_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth: float = 1e-5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the Dice loss for binary segmentation tasks.

    Parameters:
        y_pred (torch.Tensor): Predicted probabilities, expected to be logits.
        y_true (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor to avoid division by zero.
        reduction (str): Specifies the reduction to apply to the output: 'mean' or 'none'.

    Returns:
        torch.Tensor: The calculated Dice loss. Scalar if reduction is 'mean', otherwise a tensor.
    """
    y_pred = y_pred.sigmoid()
    intersection = (y_true * y_pred).sum(dim=(1, 2, 3))  # torch.sum(d, dim=(1, 2, 3))
    y_sum = y_true.sum(dim=(1, 2, 3))
    z_sum = y_pred.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (y_sum + z_sum + smooth)
    return 1 - torch.mean(dice) if reduction == "mean" else 1 - dice
