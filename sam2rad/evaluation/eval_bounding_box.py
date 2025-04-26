import argparse
import logging
import math
import os
from typing import Dict

import matplotlib.pyplot as plt
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


def build_model(config):
    """
    Choose tor build SAM or SAM2 model based on the config.
    """
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)

    return build_samrad(config)


parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
parser.add_argument("--config", required=True, help="Path to config file")

DEBUG = False


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
        self.model.prompt_sampler.p[0] = 0
        self.model.prompt_sampler.p[1] = 0
        self.model.prompt_sampler.p[2] = 1.0  # Ground truth box prompts
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
    def __init__(self, model):
        self.model = model
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
        gt = convert_to_semantic(gt)

        if DEBUG:
            boxes = boxes.cpu()
            plt.subplot(1, 2, 1)

            plt.imshow(img_orig[0])
            plt.imshow(gt[0].cpu(), alpha=0.5)
            plt.subplot(1, 2, 2)
            plt.imshow(img_orig[0])
            plt.imshow(pred[0].cpu(), alpha=0.5)
            plt.savefig("eval_box_debug.png")
            plt.close()
            import pdb

            pdb.set_trace()

        # Calculate dice and iou scores
        gt_onehot = self.one_hot(gt)
        pred_onehot = self.one_hot(pred)
        self.dice_metric(pred_onehot, gt_onehot)
        self.iou_metric(gt_onehot, pred_onehot)

    @torch.no_grad()
    def eval(self, dataloader):
        self.reset()
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in dataloader:
                self.eval_batch(batch)
                pbar.set_description(
                    f"Dice: {self.dice_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}, IoU: {self.iou_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}"
                )
                pbar.update(1)


if __name__ == "__main__":
    logger = logging.getLogger("sam2rad")
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

    # Initialize learnable prompts for each dataset
    class_tokens = torch.nn.Parameter(
        torch.randn(
            config.dataset.num_classes,
            config.dataset.num_tokens,
            256,
        )
        / math.sqrt(256)
    )

    learnable_prompts = {dataset_name: class_tokens}

    logger.info(f"Test dataset size: {len(tst_dl.dataset)}")

    model = SegmentationModel(config, learnable_prompts)
    model = model.to("cuda:0")
    model.eval()

    evaluator = Eval(model)

    evaluator.eval(tst_dl)

    # Prepare the output directory
    output_dir = os.path.join(
        os.path.dirname(config.sam_checkpoint),
        "eval_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "eval_results.txt")

    # Collect the per-class Dice and IoU scores
    dice_scores = evaluator.dice_metric.aggregate("none").nanmean(0)
    iou_scores = evaluator.iou_metric.aggregate("none").nanmean(0)

    # Collect the average Dice and IoU scores
    avg_dice = dice_scores.nanmean()
    avg_iou = iou_scores.nanmean()

    # Prepare the results as a formatted string
    results_str = "=== Evaluation Results ===\n\n"
    results_str += "Configuration:\n"
    results_str += "\nPer-Class Metrics:\n"

    # Format the per-class metrics
    results_str += "{:<10} {:>10} {:>10}\n".format("Class", "Dice", "IoU")
    results_str += "-" * 32 + "\n"
    for i, (dice_score, iou_score) in enumerate(zip(dice_scores, iou_scores)):
        results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format(
            f"Class {i}", dice_score.item(), iou_score.item()
        )

    # Add average metrics
    results_str += "\nAverage Metrics:\n"
    results_str += "-" * 32 + "\n"
    results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format("Average", avg_dice, avg_iou)

    logger.info(results_str)

    results_str = yaml.dump(config) + results_str

    # Topk images with the lowest Dice score
    dice_scores_per_img = evaluator.dice_metric.aggregate("none").nanmean(dim=1)
    values, indices = torch.topk(dice_scores_per_img, k=5, largest=False)
    logger.info("Top 5 images with the least Dice score:")
    results_str += "\nTop 5 images with the least Dice score:\n"
    for dice, idx in zip(values, indices):
        logger.info(f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}")
        results_str += f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}\n"

    # Write the results to the output file
    with open(output_file, encoding="utf-8", mode="w") as f:
        f.write(results_str)
        f.write("\n")

    logger.info("Results saved to %s" % output_file)
