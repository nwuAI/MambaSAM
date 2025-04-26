# Description: this file contains builder commonly used dataset structures.
# It accepts a list of images, ground truth files and a config file and returns a Dataset object that can be used by a dataloader.

import random
from functools import partial
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from kornia.constants import DataKey, Resample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import io

from .registry import register_dataset
from .utils import ResizeLongestSide, pad


class CenterCrop:
    """
    CenterCrop class to crop an image and its mask centered on the mask.
    """

    def __init__(self, scale: Tuple[float, float], p=0.5) -> None:
        self.scale = scale
        self.p = p

    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return img, mask

        *_, H, W = img.shape

        # Randomly sample a scale factor
        scale = np.random.uniform(*self.scale)

        # Compute the center of the mask
        y_indices, x_indices = torch.nonzero(mask[0, 0] > 0, as_tuple=True)
        if len(x_indices) == 0:
            x_center = W / 2
            y_center = H / 2
            # If the mask is empty, default to the center of the image
            x_center = float(x_indices.float().mean())
            y_center = float(y_indices.float().mean())
        else:
            x_center = x_indices.float().mean()
            y_center = y_indices.float().mean()

        # Compute crop size
        new_H = int(H * scale)
        new_W = int(W * scale)

        # Calculate the top-left and bottom-right coordinates of the crop
        x1 = x_center - new_W / 2
        y1 = y_center - new_H / 2
        x2 = x1 + new_W
        y2 = y1 + new_H

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Crop the image and mask
        mask = mask[:, :, int(y1) : int(y2), int(x1) : int(x2)]
        img = img[:, :, int(y1) : int(y2), int(x1) : int(x2)]
        return img, mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_files: List[PosixPath],
        gt_files: List[PosixPath],
        config: Dict,
        mode="Train",
    ):
        self.img_files = [str(f) for f in img_files]
        self.gt_files = [str(f) for f in gt_files]

        print(
            f"Found {len(self.img_files)} {'training' if mode == 'Train' else 'validation'} images and {len(self.gt_files)} masks."
        )

        self.img_size = config["image_size"]

        self.resize = ResizeLongestSide(self.img_size)
        self.pad = partial(pad, target=(self.img_size, self.img_size))

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(
                degrees=90, translate=(0.2, 0.2), shear=0, p=0.3, align_corners=False
            ),
            K.augmentation.RandomHorizontalFlip(p=0.3),
            # K.augmentation.RandomPerspective(
            #     distortion_scale=0.1, p=0.1, align_corners=False
            # ),
            K.augmentation.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                p=0.3,
                align_corners=False,
            ),
            data_keys=["input", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}
            },
            random_apply=(2,),
        )

        self.tst_aug = K.augmentation.AugmentationSequential(
            K.augmentation.Resize(
                size=(self.img_size, self.img_size), align_corners=False
            ),
            data_keys=["input", "mask"],
        )

        self.center_crop = CenterCrop(scale=(0.8, 1.0), p=0.5)

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.mode = mode
        self.num_classes = config["num_classes"]
        self.label_to_class_id = config.get("label_to_class_id")

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        return io.read_image(path)

    def remap_labels(self, gt):
        """
        This function is designed to modify the labels in a given tensor, `gt`, according to a predefined mapping. It's particularly useful for when the raw labels are not ordered.

        Parameters:
        - gt (torch.Tensor): The input tensor containing the original labels.

        Returns:
        torch.Tensor: The tensor with specified labels remapped to the target label.
        """

        if hasattr(self, "label_to_class_id") and self.label_to_class_id is not None:
            for k, v in self.label_to_class_id.items():
                gt[gt == k] = v

        return gt

    def get_boxes(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts non-zero areas of a mask into bounding boxes with random jitter.
        Parameters:
        - mask (Tensor): A tensor of shape (B, H, W).

        Returns:
        - Tensor: A tensor of shape (B, 4). Bounding boxes derived from the non-zero areas of the mask.
        """
        boxes = self.masks_to_boxes(mask)
        boxes += (
            torch.rand_like(boxes) * 2 - 1
        ) * 2  # Add noise to box coordinates, jitter range: [-2, 2] pixels
        return boxes

    @staticmethod
    def non_zero_coordinates(mask: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Returns coordinates for selected points based on the mask.

        - mask (Tensor): A binary mask of shape (B, H, W).
        Parameters:

        Returns:
        - Tensor: A tensor of shape (B, num_points, 2) containing the (x, y) coordinates of selected points in the batch.
        """

        batch_size = mask.size(0)
        points = []

        for i in range(batch_size):
            non_zero_points = torch.nonzero(mask[i], as_tuple=False)
            non_zero_points = non_zero_points[
                torch.randint(non_zero_points.size(0), (num_points,))
            ]
            # (row, col) -> (col, row)
            non_zero_points = non_zero_points.flip(1)
            points.append(non_zero_points)

        _points = torch.stack(points)
        return _points

    def sample_points(
        self,
        mask: torch.Tensor,
        num_foreground: int = 10,
        num_background: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts foreground and background points from a mask.

        Parameters:
        - mask (Tensor): A tensor of shape (B, 1, H, W).

        Returns:
        - points (Tensor): A tensor of shape (B, N, 2) containing the coordinates of points.
        - labels (Tensor): A tensor of shape (B, N) containing the labels of the points.
        """

        # Sample up to N foreground, background points from mask
        num_foreground = random.randint(1, num_foreground)
        # choose foreground points
        foreground = self.non_zero_coordinates(mask, num_foreground)
        # Choose background points
        background = self.non_zero_coordinates(~mask, num_background)
        labels = torch.cat([torch.ones(num_foreground), torch.zeros(num_background)])
        # Repeat the labels for each batch
        labels = labels.unsqueeze(0).repeat(mask.size(0), 1)
        return torch.cat([foreground, background], dim=1), labels

    @staticmethod
    def masks_to_boxes(masks):
        """
        Converts masks (possibly empty) into bounding boxes.

        Parameters:
        - mask (Tensor): A tensor of shape (B, H, W).

        Returns:
        - Tensor: A tensor of shape (B, 4). Bounding boxes derived from the non-zero areas of the mask.
        """
        n = masks.shape[0]
        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < 10:  # emtpy mask
                continue
            y, x = torch.where(mask != 0)

            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes

    def __getitem__(self, index):
        img = self.read_image(self.img_files[index]).float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        #
        # # 检查 gt 是否全为 0，如果是则跳过这张图片
        # if torch.all(gt == 0):
        #     return self.__getitem__((index + 1) % len(self.img_files))

        # print(f"Step 1 - Raw mask shape: {gt.shape}, unique values: {gt.unique()}")
        gt = self.remap_labels(gt).float()
        # one hot的加上，不是就去掉
        gt = torch.where(gt == 255, torch.tensor(1.0), gt)
        if self.trn_aug is not None and self.mode == "Train":
            img, gt = self.trn_aug(img[None], gt[None])
            img, gt = self.center_crop(img, gt)
        elif self.tst_aug is not None:
            img, gt = self.tst_aug(img[None], gt[None])
        # remove batch dimension
        img, gt = img[0], gt[0]
        # print(f"Step 3 - After augmentation shape: {gt.shape}, unique values: {gt.unique()}")

        # Resize to a square image
        img = self.resize(img[None], order=1)
        # print(f"Step 4 - After image resize shape: {img.shape}")

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.pad(img)

        gt = self.pad(self.resize(gt[None], order=0)).type(torch.uint8)
        # print(f"Step 5 - After mask resize and pad shape: {gt.shape}, unique values: {gt.unique()}")

        # remove batch dimension
        img, gt = img[0], gt[0]

        # convert to one-hot
        gt = F.one_hot(gt[0].long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print(f"Step 6 - After one-hot encoding shape: {gt.shape}, unique values: {gt.unique()}")

        # remove background class
        gt = gt[1:]  # (C, H, W)
        # print(f"Step 7 - After removing background class shape: {gt.shape}, unique values: {gt.unique()}")

        boxes = self.masks_to_boxes(gt)

        *_, h, w = gt.shape
        low_res_mask = F.interpolate(
            gt.float().unsqueeze(1),
            (h // 4, w // 4),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        sample = {
            "images": img,
            "masks": gt,
            "low_res_masks": low_res_mask,
            # "category_id": label,
            "boxes": boxes,
            "boxes_normalized": boxes / torch.Tensor([w, h, w, h]),
            # "points": points,
            "ignore": boxes.sum(1) == 0,  # Padding boxes,
            "filename": self.img_files[index],
        }

        return sample

    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        # 假设 mask_file 是类似 "case00_segment_1.png" 的文件名
        return mask_file.replace("gts", "imgs").replace("_segmentation_","_")
        # return mask_file.replace.replace("gts", "imgs")

    @staticmethod
    def split_train_test(file_names, test_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return train_test_split(file_names, test_size=test_size)

    @classmethod
    def from_path(cls, config, mode="Train"):
        path = Path(config["root"])
        mask_file_names = [f for f in path.glob(f"{mode}/gts/*")]
        mask_file_names.sort()

        if mode == "Train":
            trn_mask_file_names, val_mask_file_names = cls.split_train_test(
                mask_file_names, test_size=1 - config.split, seed=config.get("seed", 0)
            )

            trn_img_file_names = [
                cls.get_corresponding_image_name(str(f)) for f in trn_mask_file_names
            ]

            val_img_file_names = [
                cls.get_corresponding_image_name(str(f)) for f in val_mask_file_names
            ]

            return (
                cls(trn_img_file_names, trn_mask_file_names, config, mode="Train"),
                cls(val_img_file_names, val_mask_file_names, config, mode="Val"),
            )

        # Test
        img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in mask_file_names
        ]

        return cls(img_file_names, mask_file_names, config, mode="Test")


@register_dataset("default_test_dataset")
class TestDataset(BaseDataset):
    """
    Returns images without any augmentation.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long().squeeze(0)
        gt = F.one_hot(gt, num_classes=self.num_classes + 1).permute(
            2, 0, 1
        )  # (C+1, H, W)
        # remove background class
        gt = gt[1:]  # (C, H, W)

        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[-2:]
        img = self.pad(img)[0]
        boxes = self.masks_to_boxes(gt)

        if gt.ndim == 3:
            gt = gt.squeeze(0)

        return (
            img_orig.permute(1, 2, 0),
            img,
            gt,
            input_size,
            boxes,
            self.img_files[index],
        )


def get_dataloaders(args: dict, dataset: Dataset) -> DataLoader:
    """
    Returns a DataLoader for the given dataset with specified arguments.

    Args:
        args (dict): A dictionary containing DataLoader parameters such as
                     'batch_size' and 'num_workers'.
        dataset (Dataset): The dataset to load.

    Returns:
        DataLoader: A DataLoader configured with the provided arguments.
    """

    return DataLoader(
        dataset,
        batch_size=args.get("batch_size", 4),
        shuffle=True,
        num_workers=args.get("num_workers", 4),
        drop_last=True,
    )