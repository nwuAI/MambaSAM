import argparse
import logging
import math
import os
import sys
from typing import Dict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.metrics import DiceMetric, MeanIoU
from tqdm import tqdm

from sam2rad import (
    DATASETS,
    DotDict,
    build_sam2rad,
    build_samrad,
    convert_to_semantic,
)

logger = logging.getLogger("sam2rad")

def build_model(config):
    """
    Choose to build SAM or SAM2 model based on the config.
    """
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)

    return build_samrad(config)


parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
parser.add_argument("--config", required=True, help="Path to config file")

class SegmentationModel(torch.nn.Module):
    """
    Combines segment anything with learnable prompts.
    """

    def __init__(
        self,
        config,
        prompts: Dict[str, torch.nn.Parameter],
    ):
        super(SegmentationModel, self).__init__()
        assert "sam_checkpoint" in config, "SAM checkpoint is required."
        self.model = build_model(config)
        logger.info(self.model)
        self.dataset_names = list(prompts.keys())
        self.num_classes = list(prompts.values())[0].shape[0]
        self.learnable_prompts = torch.nn.ParameterDict(prompts)
        self.model.prompt_sampler.p[0] = 1.0  # Learnable prompts
        self.model.prompt_sampler.p[1] = 0
        self.model.prompt_sampler.p[2] = 0.0  # Ground truth box prompts
        self.model.prompt_sampler.p[3] = 0

    def forward(self, batch, dataset_index):
        """Get the learnable prompts for the dataset and make predictions"""
        prompts = self.learnable_prompts[
            self.dataset_names[dataset_index]
        ]  # (num_classes, num_tokens, 256)

        outputs = self.model(batch, prompts, inference=False)
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class Eval:
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.image_size = (
            self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
        )
        self.dice_score = 0.0
        self.iou_score = 0.0
        self.count = 0

        self.num_classes = self.model.num_classes
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
            ignore_empty=True,
        )
        self.iou_metric = MeanIoU(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
            ignore_empty=True,
        )

    def one_hot(self, masks):
        return (
            F.one_hot(masks.long(), num_classes=self.model.num_classes + 1)
            .permute(0, 3, 1, 2)
            .float()
        )

    def reset(self):
        self.dice_metric.reset()
        self.iou_metric.reset()

    @staticmethod
    def post_process(pred, input_size, original_size):
        pred = pred[:, :, : input_size[0], : input_size[1]]
        return F.interpolate(pred, original_size, mode="bilinear", align_corners=False)

    def eval_batch(self, batch):
        img_orig, images, gt, input_size, boxes, filename = batch

        # 如果 filename 是列表，取第一个元素
        if isinstance(filename, list):
            filename = filename[0]

        images, gt, boxes = (
            images.to(self.model.device),
            gt.to(self.model.device),
            boxes.to(self.model.device).view(-1, 4),
        )
        _, num_classes, *original_size = gt.shape
        w_factor, h_factor = (
            input_size[1] / original_size[1],
            input_size[0] / original_size[0],
        )
        boxes_input = boxes * torch.tensor(
            [w_factor, h_factor, w_factor, h_factor], device=boxes.device
        )
        outputs = self.model({"images": images, "masks": gt, "boxes": boxes_input}, 0)
        pred = outputs["pred"]
        pred = self.post_process(pred, input_size, original_size).view(
            -1, num_classes, *original_size
        )
        pred = convert_to_semantic(pred)

        # 将预测结果保存为 PNG 图像
        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)  # 转为 NumPy 格式并转换为 uint8
        mask_filename = os.path.join(self.output_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")
        pred_img = Image.fromarray(pred_np * 255)  # 假设预测为二值（0 和 1），将其缩放到 0-255
        pred_img.save(mask_filename)

        # 计算 Dice 和 IoU 分数
        gt = convert_to_semantic(gt)
        gt_onehot = self.one_hot(gt)
        pred_onehot = self.one_hot(pred)
        self.dice_metric(pred_onehot, gt_onehot)
        self.iou_metric(gt_onehot, pred_onehot)

    @torch.no_grad()
    def eval(self, dataloader):
        self.reset()
        os.makedirs(self.output_dir, exist_ok=True)
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in dataloader:
                self.eval_batch(batch)
                pbar.set_description(
                    f"Dice: {self.dice_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}, IoU: {self.iou_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}"
                )
                pbar.update(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    with open(args.config) as f:
        config = DotDict.from_dict(yaml.safe_load(f))

    dataset_name = config.inference.name
    ds = DATASETS.get(dataset_name, DATASETS["default_test_dataset"]).from_path(
        config.dataset, mode="Test"
    )

    tst_dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,  # Images can be of different size
        num_workers=config.get("num_workers", 1),
        pin_memory=True,
        shuffle=False,
    )

    class_tokens = torch.nn.Parameter(
        torch.randn(
            config.dataset.num_classes,
            config.dataset.num_tokens,
            256,
        )
        / math.sqrt(256)
    )

    learnable_prompts = {config.dataset.name: class_tokens}
    logger.info(f"Test dataset size: {len(tst_dl.dataset)}")
    model = SegmentationModel(config, learnable_prompts)
    checkpoint = torch.load(config.inference.model_checkpoint, map_location="cpu")
    epoch = checkpoint["epoch"]
    checkpoint = checkpoint["state_dict"]
    checkpoint = {k[len("model.") :]: v for k, v in checkpoint.items()}
    logger.info(model.load_state_dict(checkpoint))

    model = model.to("cuda:0")
    model.eval()

    output_dir = config.inference.output_dir
    evaluator = Eval(model, output_dir)
    evaluator.eval(tst_dl)
