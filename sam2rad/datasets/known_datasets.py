# Description: this file registers commonly used datasets by their names.
import warnings

import kornia as K
import torch
import torch.nn.functional as F
from torchvision import io

from . import BaseDataset, TestDataset
from .registry import register_dataset

warnings.filterwarnings("ignore")


@register_dataset("default_segmentation")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = None


@register_dataset("shoulder")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.2),
            K.augmentation.RandomVerticalFlip(p=0.2),
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("wrist")
class WristScans(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("hip")
class Hip(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("MSD")
class MSDDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Apply augmentations during training
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        # Remap 0 -> background, 255 -> thyroid region (1)
        return (gt > 0).long()  # Return 1 for thyroid, 0 for background

@register_dataset("tn3k")
class ThyroidUltrasoundDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Apply augmentations during training
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        # Remap 0 -> background, 255 -> thyroid region (1)
        return (gt > 0).long()  # Return 1 for thyroid, 0 for background


@register_dataset("promise12")
class Promise12Dataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Apply augmentations during training
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )


    def remap_labels(self, gt):
        # Remap 0 -> background, 255 -> thyroid region (1)
        return (gt > 0).long()  # Return 1 for thyroid, 0 for background

@register_dataset("monuseg")
class MonusegDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Apply augmentations during training


    def remap_labels(self, gt):
        # Remap 0 -> background, 255 -> thyroid region (1)
        return (gt > 0).long()  # Return 1 for thyroid, 0 for background


@register_dataset("isic2016")
class ISIC2016Dataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Apply augmentations during training
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        # Remap 0 -> background, 255 -> thyroid region (1)
        return (gt > 0).long()  # Return 1 for thyroid, 0 for background
@register_dataset("3dus_chop")
class Hip3DChopDataset(BaseDataset):
    # Map RGB values to class IDs
    label_to_class_id = {
        (255, 0, 0): 1,  # Red -> Class 0
        (0, 255, 0): 2,  # Green -> Class 1
        (0, 0, 255): 3,  # Blue -> Class 2
    }

    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    @staticmethod
    def remap_labels(gt: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Remap RGB labels to class IDs.
        Assumes `gt` is a tensor of shape (3, H, W) or (
        H, W, 3) where each pixel is an RGB tuple.
        """
        # If the input is in shape (3, H, W), we need to permute it to (H, W, 3)
        if gt.shape[0] == 3:
            gt = gt.permute(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)

        # Now proceed with the original logic
        class_map = torch.zeros(gt.shape[:2], dtype=torch.long)
        for color, class_id in Hip3DChopDataset.label_to_class_id.items():
            match = (gt == torch.tensor(color, dtype=gt.dtype)).all(dim=-1)
            class_map[match] = class_id
        return class_map


@register_dataset("3dus_chop_test")
class ChopTestDataset(Hip3DChopDataset):
    """
    Returns images without any augmentation.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]
        # convert to one-hot
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(
            2, 0, 1
        )  # (C+1, H, W)
        # remove background class
        gt = gt[1:]  # (C, H, W)

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

@register_dataset("synapse")
class SynapseDataset(BaseDataset):
    """
    Synapse 数据集，处理 mask，转换为类别 ID，并支持数据增强
    """

    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # 数据增强
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )




@register_dataset("synapse_test")
class SynapseTestDataset(SynapseDataset):
    """
    Synapse 测试集，不进行数据增强
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)  # 确保 gt 形状为 (H, W)
        gt = F.one_hot(gt, num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # 移除背景类别
        gt = gt[1:]
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[-2:]
        img = self.pad(img)[0]
        boxes = self.masks_to_boxes(gt)
        return (
            img_orig.permute(1, 2, 0),
            img,
            gt,
            input_size,
            boxes,
            self.img_files[index],
        )

@register_dataset("monuseg_test")
class MonusegTestDataset(ThyroidUltrasoundDataset):
    """
    Returns images without any augmentation.
    This is the test version of the TN3K dataset where no data augmentation is applied.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        # print("gt shape after read_mask:", gt.shape)  # 检查 read_mask 的输出

        gt = self.remap_labels(
            gt).long()  # Convert ground truth to long tensor (0 for background, 1 for thyroid region)
        # print("gt shape after remap_labels:", gt.shape)

        # 确保 gt 是 2D Tensor (H, W)
        if gt.ndim > 2:
            gt = gt.squeeze(0)  # 去掉额外的维度
        # print("gt shape after squeezing:", gt.shape)

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        # 转为 one-hot 编码
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print("gt shape after one_hot:", gt.shape)

        # Remove background class
        gt = gt[1:]  # (C, H, W)
        # print("gt shape after removing background:", gt.shape)

        # Get bounding boxes from the ground truth mask
        boxes = self.masks_to_boxes(gt)

        return (
            img_orig.permute(1, 2, 0),  # Original image (H, W, C)
            img,  # Resized image
            gt,  # Ground truth mask
            input_size,  # Input size of the image
            boxes,  # Bounding boxes for the thyroid region
            self.img_files[index],  # Filename of the image
        )


@register_dataset("acdc")
class ACDCDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(
                degrees=30, translate=(0.2, 0.2), shear=0, p=0.5, align_corners=False
            ),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
            random_apply=(2,),
        )

    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask


@register_dataset("acdc_test")
class ACDCTestDataset(TestDataset):
    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask

@register_dataset("promise12_test")
class Promise12TestDataset(ThyroidUltrasoundDataset):
    """
    Returns images without any augmentation.
    This is the test version of the TN3K dataset where no data augmentation is applied.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        # print("gt shape after read_mask:", gt.shape)  # 检查 read_mask 的输出

        gt = self.remap_labels(
            gt).long()  # Convert ground truth to long tensor (0 for background, 1 for thyroid region)
        # print("gt shape after remap_labels:", gt.shape)

        # 确保 gt 是 2D Tensor (H, W)
        if gt.ndim > 2:
            gt = gt.squeeze(0)  # 去掉额外的维度
        # print("gt shape after squeezing:", gt.shape)

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        # 转为 one-hot 编码
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print("gt shape after one_hot:", gt.shape)

        # Remove background class
        gt = gt[1:]  # (C, H, W)
        # print("gt shape after removing background:", gt.shape)

        # Get bounding boxes from the ground truth mask
        boxes = self.masks_to_boxes(gt)

        return (
            img_orig.permute(1, 2, 0),  # Original image (H, W, C)
            img,  # Resized image
            gt,  # Ground truth mask
            input_size,  # Input size of the image
            boxes,  # Bounding boxes for the thyroid region
            self.img_files[index],  # Filename of the image
        )


@register_dataset("tn3k_test")
class ThyroidUltrasoundTestDataset(ThyroidUltrasoundDataset):
    """
    Returns images without any augmentation.
    This is the test version of the TN3K dataset where no data augmentation is applied.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        # print("gt shape after read_mask:", gt.shape)  # 检查 read_mask 的输出

        gt = self.remap_labels(
            gt).long()  # Convert ground truth to long tensor (0 for background, 1 for thyroid region)
        # print("gt shape after remap_labels:", gt.shape)

        # 确保 gt 是 2D Tensor (H, W)
        if gt.ndim > 2:
            gt = gt.squeeze(0)  # 去掉额外的维度
        # print("gt shape after squeezing:", gt.shape)

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        # 转为 one-hot 编码
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print("gt shape after one_hot:", gt.shape)

        # Remove background class
        gt = gt[1:]  # (C, H, W)
        # print("gt shape after removing background:", gt.shape)

        # Get bounding boxes from the ground truth mask
        boxes = self.masks_to_boxes(gt)

        return (
            img_orig.permute(1, 2, 0),  # Original image (H, W, C)
            img,  # Resized image
            gt,  # Ground truth mask
            input_size,  # Input size of the image
            boxes,  # Bounding boxes for the thyroid region
            self.img_files[index],  # Filename of the image
        )


@register_dataset("isic2016_test")
class ISIC2016TestDataset(ThyroidUltrasoundDataset):
    """
    Returns images without any augmentation.
    This is the test version of the TN3K dataset where no data augmentation is applied.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        # print("gt shape after read_mask:", gt.shape)  # 检查 read_mask 的输出

        gt = self.remap_labels(
            gt).long()  # Convert ground truth to long tensor (0 for background, 1 for thyroid region)
        # print("gt shape after remap_labels:", gt.shape)

        # 确保 gt 是 2D Tensor (H, W)
        if gt.ndim > 2:
            gt = gt.squeeze(0)  # 去掉额外的维度
        # print("gt shape after squeezing:", gt.shape)

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        # 转为 one-hot 编码
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print("gt shape after one_hot:", gt.shape)

        # Remove background class
        gt = gt[1:]  # (C, H, W)
        # print("gt shape after removing background:", gt.shape)

        # Get bounding boxes from the ground truth mask
        boxes = self.masks_to_boxes(gt)

        return (
            img_orig.permute(1, 2, 0),  # Original image (H, W, C)
            img,  # Resized image
            gt,  # Ground truth mask
            input_size,  # Input size of the image
            boxes,  # Bounding boxes for the thyroid region
            self.img_files[index],  # Filename of the image
        )

@register_dataset("MSD_test")
class ThyroidUltrasoundTestDataset(ThyroidUltrasoundDataset):
    """
    Returns images without any augmentation.
    This is the test version of the TN3K dataset where no data augmentation is applied.
    """

    def __getitem__(self, index):
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        # print("gt shape after read_mask:", gt.shape)  # 检查 read_mask 的输出

        gt = self.remap_labels(
            gt).long()  # Convert ground truth to long tensor (0 for background, 1 for thyroid region)
        # print("gt shape after remap_labels:", gt.shape)

        # 确保 gt 是 2D Tensor (H, W)
        if gt.ndim > 2:
            gt = gt.squeeze(0)  # 去掉额外的维度
        # print("gt shape after squeezing:", gt.shape)

        # Normalize image
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        # 转为 one-hot 编码
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(2, 0, 1)  # (C+1, H, W)
        # print("gt shape after one_hot:", gt.shape)

        # Remove background class
        gt = gt[1:]  # (C, H, W)
        # print("gt shape after removing background:", gt.shape)

        # Get bounding boxes from the ground truth mask
        boxes = self.masks_to_boxes(gt)

        return (
            img_orig.permute(1, 2, 0),  # Original image (H, W, C)
            img,  # Resized image
            gt,  # Ground truth mask
            input_size,  # Input size of the image
            boxes,  # Bounding boxes for the thyroid region
            self.img_files[index],  # Filename of the image
        )