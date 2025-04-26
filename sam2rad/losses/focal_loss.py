from torchvision.ops import sigmoid_focal_loss


def average_spatial(inputs, target, reduction="mean", *args, **kwargs):
    """Averge focal loss along the spatial dimensions."""
    loss = sigmoid_focal_loss(inputs, target, reduction=reduction, *args, **kwargs)
    loss = loss.mean(dim=(1, 2, 3))

    return loss


focal_loss = average_spatial
