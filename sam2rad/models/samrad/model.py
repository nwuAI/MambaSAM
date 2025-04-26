from typing import List, Tuple

import torch
import torch.nn.functional as F

from sam2rad.models import Model as BaseModel
from sam2rad.models.sam.modeling import ImageEncoderViT, MaskDecoder


class Model(BaseModel):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        mask_decoder: MaskDecoder,
        prompt_sampler,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_sampler = prompt_sampler

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.image_size = image_encoder.img_size

    def forward(self, batch, learnable_prompts, inference=False):
        """
        Parameters:
        - learnable_prompts: Tensor of shape (batch size|num_classes|batch size*num_classes, num_prompts, 256).
        - batch: dataloader batch.
        - masks: Optional tensor of shape (B, 1, H, W); can be None during inference.
        """

        image = batch["images"]
        masks = batch.get("masks", None)
        assert inference or masks is not None, "Masks must be provided during training."
        image_embedding, high_res_features = self.image_encoder(
            image
        )  # (B, 256, H//16, W//16)
        # Print where image_encoder is coming from
        # print(f"ImageEncoderViT class: {self.image_encoder.__class__.__name__}")
        # print(f"ImageEncoderViT source file: {self.image_encoder.__class__.__module__}")
        image_pe = self.prompt_sampler.prompt_encoder.get_dense_pe().detach()
        # If not evaluating the manual prompts, use the learnable prompts
        # If there are batched prompts per image, repeat the image embedding
        batch_size = image_embedding.shape[0]
        num_classes = learnable_prompts.shape[0]
        learnable_prompts = learnable_prompts.repeat(batch_size, 1, 1)
        image_embedding = image_embedding.repeat_interleave(num_classes, dim=0)
        for i, feat in enumerate(high_res_features):
            high_res_features[i] = feat.repeat_interleave(num_classes, dim=0)

        if inference:  # Use predicted prompts only
            sparse_embeddings, dense_embeddings, interim_mask_output, pred_boxes = (
                self.prompt_sampler.prompt_learner(
                    image_features=high_res_features + [image_embedding],
                    queries=learnable_prompts,
                )
            )

            # Upsample to original image size
            interim_mask_output = F.interpolate(
                interim_mask_output, scale_factor=4, mode="bilinear"
            )

        else:  # Use both predicted and manual prompts
            prompt_outputs = self.prompt_sampler(
                image_features=high_res_features + [image_embedding],
                learnable_prompts=learnable_prompts,
                batch=batch,
            )

            sparse_embeddings = prompt_outputs["sparse_embeddings"]
            interim_mask_output = prompt_outputs["interim_mask_output"]
            pred_boxes = prompt_outputs["pred_boxes"]
            dense_embeddings = prompt_outputs["dense_embeddings"]

            if interim_mask_output is not None:
                interim_mask_output = F.interpolate(
                    interim_mask_output, scale_factor=4, mode="bilinear"
                )

        # Decode mask
        low_res_masks, iou_pred, object_score_logits = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, H//16, W//16)
            image_pe=image_pe,  # (1, 256, H//16, W//16)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, N, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, H//16, W//16)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        masks = F.interpolate(
            low_res_masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": masks,
            "iou_predictions": iou_pred,
            "object_score_logits": object_score_logits,
            "pred_boxes": pred_boxes,
            "interim_mask_output": interim_mask_output,  # (B, 1, H, W)
            # "learned_embeddings": _learned_embeddings,  # (B, 2/0, 256)
            # "box_embeddings": box_embeddings,  # (B,  2/0, 256)
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
