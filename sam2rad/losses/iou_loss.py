import torch


def iou_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float = 1.0, beta: float = 1.0, smooth: float = 1e-5, reduction="mean") -> torch.Tensor:
    """
    Computes the Tversky loss for binary segmentation tasks.

    Tversky loss is a generalization of Dice loss that adds flexibility in controlling the penalty for false positives and false negatives.

    Parameters:
        y_pred (torch.Tensor): Predicted segmentation maps, expected to be logits.
        y_true (torch.Tensor): Ground truth segmentation maps.
        alpha (float): Weight for false negatives.
        beta (float): Weight for false positives.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Scalar tensor representing the Tversky loss.
    """
    y_pred = torch.sigmoid(y_pred)
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_index = tp / (tp + alpha * fn + beta * fp + smooth)
    # CAUTION: If the foreground is empty and the prediction is also empty, the loss should be 0 not 1.

    empty = (y_true.sum(dim=(1, 2, 3)) < 10) & ((y_pred > 0.5).sum(dim=(1, 2, 3)) < 10)
    tversky_index[empty] = 1 - tversky_index[empty] # Overlap: 0% -> 100%

    if reduction == "mean":
        return 1 - torch.mean(tversky_index)
    elif reduction == "sum":
        return 1 - torch.sum(tversky_index)
    else:
        return 1 - tversky_index
