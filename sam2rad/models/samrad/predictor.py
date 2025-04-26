from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sam2rad.models.samrad.model import Model


class Sam2RadPredictor:
    def __init__(
        self,
        model: Model,
        pred_iou_thresh: float = 0.0,
        mask_threshold: float = 0.0,
        stability_score_thresh: float = 0.0,
        stability_score_offset: float = 0.2,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          model : The model to use for mask prediction.
        """
        super().__init__()
        self.model = model
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.mask_threshold = mask_threshold
        self.stability_score_offset = stability_score_offset
        self.transform = ResizeLongestSide(self.model.image_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]

        self.set_torch_image(input_image_torch, image.shape[:2])

    def preprocess_mask(self, mask: np.array) -> np.array:
        """
        Process a mask to the form expected by self.model.prompt_sampler.prompt_encoder._embed_mask
        """
        # The size to transform the mask is determined by the image encoder
        # The image should be set first.
        assert self.is_image_set, "An image must be set before a mask can be processed."

        # Resize and pad the mask to the input size of the model (i.e., image_size // 4)
        # resize longest side
        input_mask = np.array(
            resize(
                to_pil_image(mask),
                (self.input_size[0] // 4, self.input_size[1] // 4),
                interpolation=0,
            )
        )  # (H, W)
        # pad the mask to a square input
        h, w = input_mask.shape
        padh = self.model.image_encoder.img_size // 4 - h
        padw = self.model.image_encoder.img_size // 4 - w
        input_mask = np.pad(
            input_mask,
            ((0, padh), (0, padw)),
            mode="constant",
            constant_values=0,
        )  # (H, W)
        # Add batch dimension
        input_mask = input_mask[None]  # (1, H, W)
        return input_mask

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        # import pdb; pdb.set_trace()
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict_prompts(self, learnable_prompts):
        # Predicted box and mask prompts
        image_embedding = self.features
        sparse_embeddings, dense_embeddings = self.model.prompt_sampler.prompt_learner(
            image_embedding, learnable_prompts
        )
        dense_embeddings = F.interpolate(
            dense_embeddings, scale_factor=4, mode="bilinear"
        )
        mask_output_classifier = self.model.classifier(dense_embeddings)
        dense_embeddings = self.model.prompt_sampler.prompt_encoder._embed_masks(
            mask_output_classifier
        )
        mask_output_classifier = F.interpolate(
            dense_embeddings, scale_factor=4, mode="bilinear"
        )
        # Encode learned box coordinates
        pred_boxes = torch.sigmoid(
            self.model.box_regression_head(sparse_embeddings[:, :2])
        )
        learned_box_embeddings = self.model.prompt_sampler.prompt_encoder._embed_boxes(
            boxes=pred_boxes
            * self.model.prompt_sampler.prompt_encoder.input_image_size[0]
        )

        return {
            "learned_sparse_embeddings": sparse_embeddings[:, 2:],
            "box_embeddings": learned_box_embeddings,
            "dense_embeddings": dense_embeddings,
            "mask_output_classifier": mask_output_classifier,
        }

    @torch.no_grad()
    def predict_torch(
        self,
        learned_prompts: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          learned_prompts (torch.Tensor): A 1xNx256 tensor representing learned prompts of a class. N is the number of prompts for the class. learned prompts are concatenated with manual prompts and fed to the mask decoder.
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        assert any(
            p is not None for p in [learned_prompts, point_coords, boxes, mask_input]
        ), "At least one prompt type must be provided."

        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed manual prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_sampler.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Concatenate learned prompts
        image_embed = self.features
        # Predict prompts
        if learned_prompts is not None:
            if image_embed.shape[0] != learned_prompts.shape[0]:
                image_embed = torch.repeat_interleave(
                    image_embed, learned_prompts.shape[0], dim=0
                )

            learned_embeddings = self.predict_prompts(learned_prompts)
            learned_sparse_embeddings = learned_embeddings[
                "learned_sparse_embeddings"
            ]  # (B, N, 256)
            learned_box_embeddings = learned_embeddings["box_embeddings"]  # (B, 2, 256)
            dense_embeddings = learned_embeddings[
                "dense_embeddings"
            ]  # (B, 256, H//16, W//16)

            input_prompts = torch.cat(
                [learned_box_embeddings, learned_sparse_embeddings], dim=1
            )

            # Use both learned and manual prompts
            if sparse_embeddings.size(1) > 0:
                input_prompts = torch.cat([sparse_embeddings, input_prompts], dim=1)

        else:
            # No learned prompts are provided
            input_prompts = sparse_embeddings

        # Predict masks
        low_res_masks, iou_predictions, object_score_logits = self.model.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.prompt_sampler.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=input_prompts,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Remove false positives based on the predicted iou overlap
        # low_res_masks[object_score_logits < self.pred_iou_thresh] = -10
        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, object_score_logits, low_res_masks

    def predict(
        self,
        learned_prompts: Optional[torch.Tensor] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          learned_prompts (torch.Tensor): A 1xNx256 tensor representing learned prompts of a class. N is the number of prompts for the class. learned prompts are concatenated with manual prompts and fed to the mask decoder.
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """

        assert any(
            p is not None for p in [learned_prompts, point_coords, box, mask_input]
        ), "At least one prompt type must be provided."

        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = (
            None,
            None,
            None,
            None,
        )
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device
            )
            coords_torch, labels_torch = (
                coords_torch[None, :, :],
                labels_torch[None, :],
            )
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            # Resize and pad the mask to the size expected by the prompt encoder
            mask_input = self.preprocess_mask(mask_input)
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device
            )
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, object_score_logits, low_res_masks = self.predict_torch(
            learned_prompts,
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.detach().cpu().numpy()
        iou_predictions_np = iou_predictions.detach().cpu().numpy()
        low_res_masks_np = low_res_masks.detach().cpu().numpy()
        return (
            masks_np,
            iou_predictions_np,
            object_score_logits,
            low_res_masks_np,
        )

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self.features is not None
        ), "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
