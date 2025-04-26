import argparse
import logging
import math
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import termcolor
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.ops as ops
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Dice
from torchmetrics.classification import MulticlassJaccardIndex as IoU

from sam2rad import (
    DATASETS,
    AverageMeter,
    CompositeLoss,
    DotDict,
    build_sam2rad,
    build_samrad,
    convert_to_semantic,
    dice_loss,
    focal_loss,
    get_dataloaders,
    overlay_contours,
)
from sam2rad.logging import setup_logging

setup_logging(output="training_logs", name="sam2rad")

logger = logging.getLogger("sam2rad")

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--config", type=str, help="Path to model config file")


class SavePredictionsCallback(pl.Callback):
    """
    A PyTorch Lightning callback to save and visualize predictions during training and validation.
    """

    def __init__(self):
        self.val_outputs = []
        self.train_outputs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 5:
            self.train_outputs.append(outputs)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 5:
            self.val_outputs.append(outputs)

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = [o for o in self.val_outputs if o is not None and "pred_masks" in o]

        if len(outputs) == 0:
            print("⚠️ No valid outputs in validation epoch, skipping visualization")
            return
        pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
        gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
        images = torch.cat([o["images"] for o in outputs], dim=0)
        fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))
        if pred.size(0) == 1:
            axes = axes[None, ...]

        for i, (p, g, img) in enumerate(zip(pred, gt, images)):
            img_gt = overlay_contours(
                img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
            )
            img_pred = overlay_contours(
                img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
            )
            axes[i, 0].imshow(img_gt)
            axes[i, 0].imshow(g.cpu(), alpha=0.2)
            axes[i, 1].imshow(img_pred)
            axes[i, 1].imshow(p.cpu(), alpha=0.2)

        plt.savefig("debug_val_progress.png", bbox_inches="tight")
        plt.close()

        self.val_outputs.clear()

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = [o for o in self.train_outputs if o is not None and "pred_masks" in o]

        if len(outputs) == 0:
            return
        if len(outputs) > 0:
            pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
            gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
            images = torch.cat([o["images"] for o in outputs], dim=0)

            fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))
            if pred.size(0) == 1:
                axes = axes[None, ...]

            for i, (p, g, img) in enumerate(zip(pred, gt, images)):
                img_gt = overlay_contours(
                    img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
                )
                img_pred = overlay_contours(
                    img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
                )
                axes[i, 0].imshow(img_gt)
                axes[i, 0].imshow(g.cpu(), alpha=0.2)
                axes[i, 1].imshow(img_pred)
                axes[i, 1].imshow(p.cpu(), alpha=0.2)

                # Contours

            plt.savefig("debug_train_progress.png", bbox_inches="tight")
            plt.close()

            self.train_outputs.clear()

        return super().on_train_epoch_end(trainer, pl_module)


def build_model(config):
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)

    return build_samrad(config)


class SegmentationModule(torch.nn.Module):
    """
    Combines segment anything with learnable prompts.
    """

    def __init__(
        self,
        cfg,
        prompts: Dict[str, torch.nn.Parameter],
    ):
        super(SegmentationModule, self).__init__()
        self.model = build_model(cfg)
        # Sometimes use manual prompts only (box, mask, etc.) so that the predicted prompts align with manual prompts.
        self.model.prompt_sampler.p[0] = 1  # Learned prompts
        # If box or mask prompt is used during training, the model can be prompted to correct a prediction by providing a box or mask prompt (human-in-the-loop)
        self.model.prompt_sampler.p[2] = 0.5  # Box
        self.model.prompt_sampler.p[3] = 0.1  # Mask

        self.dataset_names = list(prompts.keys())
        self.learnable_prompts = torch.nn.ParameterDict(prompts)

        self.num_classes = self.learnable_prompts[cfg.dataset.name].size(0)

    def forward(self, batch, dataset_index=0, inference=False):
        """Get the learnable prompts for the dataset and make predictions"""
        imgs = batch["images"]
        prompts = self.learnable_prompts[self.dataset_names[dataset_index]].to(
            imgs.device
        )  # (num_classes, num_tokens, 256)

        outputs = self.model(batch, prompts, inference=inference)
        return outputs


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: List[float],
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)

        self.label_smoothing = 0.1
        self.image_size = (
            self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
        )

        self.train_dice_metric = AverageMeter()
        self.train_iou_metric = AverageMeter()
        self.val_dice_metric = AverageMeter()
        self.val_iou_metric = AverageMeter()

        self.iou = IoU(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

        self.dice = Dice(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

    def on_validation_epoch_end(self) -> None:
        self.val_dice_metric.reset()
        self.val_iou_metric.reset()
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        self.train_dice_metric.reset()
        self.train_iou_metric.reset()
        return super().on_train_epoch_end()

    @staticmethod
    def generalized_box_iou_loss(pred_boxes, target_boxes, ignore_boxes=None):
        """
        Generalized box iou loss.
        pred_boxes: (B, 4) x1, y1, x2, y2
        target_boxes: (B, 4) x1, y1, x2, y2
        """
        if ignore_boxes is None:
            ignore_boxes = torch.zeros_like(pred_boxes).bool()
        loss = ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction="none")
        loss = (loss * (1 - ignore_boxes)).sum() / (1 - ignore_boxes).sum()

        return loss

    @staticmethod
    def reshape_inputs(batch):
        batch["boxes"] = batch["boxes"].reshape(-1, 4)
        batch["boxes_normalized"] = batch["boxes_normalized"].reshape(-1, 4)
        batch["ignore"] = batch["ignore"].reshape(-1)
        lr_masks = batch["low_res_masks"]
        batch["low_res_masks"] = lr_masks.reshape(
            -1, 1, lr_masks.size(2), lr_masks.size(3)
        )
        masks = batch["masks"]
        batch["masks"] = masks.reshape(-1, 1, masks.size(2), masks.size(3))

        return batch

    def training_step(self, batch, batch_idx):
        b, c, h, w = batch["masks"].shape
        batch = self.reshape_inputs(batch)
        gt = batch["masks"].float()  # (B*C, 1, H, W)
        # print(gt.unique())
        outputs = self.model(batch)
        pred = outputs["pred"]
        # print(outputs["pred"])
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        # print('seg',loss_seg)
        # Compute loss for non-empty masks only
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 10).float()
        # print('is_non', is_non_empty)
        # if is_non_empty.sum() == 0:
        #     # print("⚠️ Skipping this batch because all masks are empty ⚠️")
        #     return None
        loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)
        # print('2loss_seg', loss_seg)
        # Bounding box regression loss
        loss_box = 0.0
        if outputs["pred_boxes"] is not None:
            pred_boxes = outputs["pred_boxes"]  # x1, y1, x2, y2
            target_boxes = batch["boxes_normalized"]  # x1, y1, x2, y2
            ignore_boxes = batch["ignore"].float()
            loss_box = self.generalized_box_iou_loss(
                pred_boxes, target_boxes, ignore_boxes
            )

        # Object prediction head
        object_score_logits = torch.clip(
            outputs["object_score_logits"].view(-1), -10, 10
        )
        if self.label_smoothing > 0:
            target = (
                is_non_empty * (1 - self.label_smoothing) + self.label_smoothing / 2
            )

        else:
            target = is_non_empty

        loss_object = F.binary_cross_entropy_with_logits(object_score_logits, target)

        interim_mask_loss = 0.0
        if outputs["interim_mask_output"] is not None:
            interim_mask_loss = ops.sigmoid_focal_loss(
                outputs["interim_mask_output"], gt, reduction="none", alpha=0.6, gamma=3
            )

            interim_mask_loss = interim_mask_loss.mean(dim=(1, 2, 3))
            interim_mask_loss = (interim_mask_loss * is_non_empty).sum() / (
                is_non_empty.sum() + 1e-6
            )

        train_loss = loss_seg + loss_object + loss_box + 100 * interim_mask_loss

        # Compute metrics
        _pred = pred.clone().detach()
        _pred[object_score_logits < 0] = -1
        pred_semantic = convert_to_semantic(_pred.detach().view(b, c, h, w))
        gt_semantic = convert_to_semantic(gt.view(b, c, h, w))

        self.train_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.train_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        self.log_dict(
            {
                "train_loss_seg": loss_seg,
                "interim_mask_loss": interim_mask_loss,
                "train_loss_box": loss_box,
                "train_loss_object": loss_object,
                "train_iou": self.train_iou_metric.get_avg(),
                "train_dice": self.train_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": train_loss,
            "iou": self.train_iou_metric.get_avg(),
            "dice": self.train_dice_metric.get_avg(),
            "confidence": object_score_logits,
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            # DEBUG
            "pred_boxes": outputs["pred_boxes"],
            "interim_mask_output": outputs["interim_mask_output"],
            "gt_boxes": batch["boxes_normalized"],
        }

    def denormalize(self, img):
        img = img * self.pixel_std.to(img.device) + self.pixel_mean.to(img.device)
        return (img * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        b, c, h, w = batch["masks"].shape
        batch = self.reshape_inputs(batch)
        gt = batch["masks"].float()  # (B, C, H, W)
        outputs = self.model(batch, dataloader_idx, inference=True)
        pred = outputs["pred"]
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        # train on non-empty masks only
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 1).float()
        # if is_non_empty.sum() == 0:
        #     # print("⚠️ Skipping this batch because all masks are empty ⚠️")
        #     return None
        loss_seg = (loss_seg * is_non_empty).sum() / (is_non_empty.sum() + 1e-6)
        object_score_logits = outputs["object_score_logits"].view(-1)
        loss_object = F.binary_cross_entropy_with_logits(
            object_score_logits, is_non_empty
        )

        pred[object_score_logits < 0] = -1
        pred_semantic = convert_to_semantic(pred.detach().view(b, c, h, w))
        gt_semantic = convert_to_semantic(gt.view(b, c, h, w))

        self.val_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.val_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        # log the loss and metrics
        self.log_dict(
            {
                "val_loss_seg": loss_seg,
                "val_loss_object": loss_object,
                "val_iou": self.val_iou_metric.get_avg(),
                "val_dice": self.val_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            "confidence": object_score_logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.max_epochs, eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = DotDict(config)

    # Register a custom dataset or use a default one, e.g., dataset_obj = DATASETS["default_segmentation"]
    dataset_obj = DATASETS[config.dataset.name]

    trn_ds, val_ds = dataset_obj.from_path(config.dataset)
    # Debug: faster validation
    # val_ds = torch.utils.data.Subset(val_ds, range(0, 100))

    trn_dl = get_dataloaders(config.dataset, trn_ds)
    val_dl = get_dataloaders(config.dataset, val_ds)

    logger.info(f"Train dataset size: {len(trn_dl.dataset)}")
    logger.info(f"Validation dataset size: {len(val_dl.dataset)}")

    # Initialize learnable prompts for each dataset
    class_tokens = torch.nn.Parameter(
        torch.randn(
            config.dataset.num_classes,
            config.dataset.num_tokens,
            256,
        )
        / math.sqrt(256)
    )

    loss_fn = CompositeLoss(
        [
            partial(dice_loss, reduction="none"),
            partial(focal_loss, reduction="none", alpha=0.7, gamma=3),
        ],
        weights=[1.0, 10.0],
    )

    model = SegmentationModule(config, {config.dataset.name: class_tokens})
    logger.info(model)
    termcolor.colored("Trainable parameters:", "red")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(termcolor.colored(f"{name} | {param.size()}", "red"))

    learner = Learner(model, loss_fn=loss_fn, lr=1e-4)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    wandb_logger = WandbLogger(
        project=config.get("wandb_project_name", "hfunet"),
        log_model=False,
        save_dir="./logs",
    )
    wandb_logger.watch(model, log_graph=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        dirpath="checkpoints"
        if config.get("save_path") is None
        else config.get("save_path"),
        save_last=True,
        filename="model_{epoch:02d}-{val_dice:.2f}",
        save_top_k=3,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=20,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, SavePredictionsCallback()],
        accelerator="gpu",  # run on all available GPUs
    )

    trainer.fit(
        learner,
        train_dataloaders=trn_dl,
        val_dataloaders=val_dl,
        ckpt_path=config.training.get("resume"),
    )
