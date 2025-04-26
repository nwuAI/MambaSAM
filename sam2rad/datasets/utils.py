import torch.nn.functional as F


def pad(image, target=(1024, 1024)):
    """
    Pads a grayscale image to a size of 1024x1024 pixels.

    Parameters:
    - image (np.ndarray): The input image in HW (Height, Width) format, representing a grayscale image.

    The function pads the input image to ensure its size is 1024x1024 pixels, filling any additional space with zeros. This operation maintains the original image's aspect ratio centered in the padded area.
    """

    *_, h, w = image.shape

    padh = target[0] - h
    padw = target[1] - w
    image = F.pad(image, (0, padw, 0, padh), mode="constant", value=0)
    return image


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length'.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, image, order=1):
        """
        Processes a batch of images using interpolation.

        Parameters:
        - input_tensor (torch.Tensor): A tensor of shape (B, C, H, W) representing a batch of images.

        Interpolation options:
        - order (int): Specifies the interpolation method to be used.
            - 1: Bilinear interpolation.
            - 0: Nearest neighbor interpolation.
        """

        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear" if order == 1 else "nearest"
        )

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
