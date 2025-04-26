import logging
import math

import torch
import torch.nn as nn

from sam2rad.models.sam.modeling.mask_decoder import (
    MaskDecoder as SAMMaskDecoderImpl,
)
from sam2rad.models.sam.modeling.transformer import TwoWayTransformer

from .base import MaskDecoder, MaskDecoderFactory
from .registry import register_mask_decoder

logger = logging.getLogger("sam2rad")


class LoRAqkv(nn.Module):
    """
    Applies low-rank adaptation to a linear projection.
    """

    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.proj(x) + self.w_b(self.w_a(x))
        return x


# Mask decoder variants
class SAMMaskDecoder(MaskDecoder):
    """
    SAM mask decoder for full-finetuning or frozen evaluation.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = []

        for name, param in self.net.named_parameters():
            if any((_train in name) for _train in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def predict_masks(self, *args, **kwargs):
        return self.net.predict_masks(*args, **kwargs)

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Extract the mask decoder state dict from model checkpoint
        state_dict = {
            k.replace("mask_decoder.", ""): v
            for k, v in state_dict.items()
            if "mask_decoder" in k
        }
        ignore_keys = [
            "obj_score_token",
            "pred_obj_score_head",
        ]  # Parameters not present in SAM mask decoder

        missing_keys, unexpected_keys = self.net.load_state_dict(
            state_dict, strict=False
        )

        missing_keys = {
            k for k in missing_keys if not any([key in k for key in ignore_keys])
        }

        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()

        if unexpected_keys:
            logger.error(unexpected_keys)
            raise RuntimeError()

        logger.info(
            "%s loaded checkpoint from %s successfully.",
            self.net.__class__.__name__,
            checkpoint_path,
        )


class LoRAMaskDecoder(MaskDecoder):
    """Applies low-rank adaptation to a SAM's mask decoder."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = [
            "q_proj.w_a.weight",
            "q_proj.w_b.weight",
            "v_proj.w_a.weight",
            "v_proj.w_b.weight",
            "pred_obj_score_head",
            "obj_score_token",
            "mask_tokens",
            "output_upscaling",
            "output_hypernetworks_mlps",
        ]

        for name, param in self.net.named_parameters():
            if any((_train in name) for _train in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")  # ["model"]
        # Extract the mask decoder state dict from model checkpoint
        state_dict = {
            k.replace("mask_decoder.", ""): v
            for k, v in state_dict.items()
            if "mask_decoder" in k
        }
        new_state_dict = {}

        for k in state_dict:
            # remap keys '.....q_proj....' -> '.....q_proj.proj....'
            # '.....v_proj....' -> '.....v_proj.proj....'
            if "q_proj" in k:
                new_state_dict[k.replace("q_proj", "q_proj.proj")] = state_dict[k]

            elif "v_proj" in k:
                new_state_dict[k.replace("v_proj", "v_proj.proj")] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        state_dict = new_state_dict

        # Sanity check
        for k in state_dict:
            if k not in self.net.state_dict().keys():
                logger.error(f"Key {k} not found in model state_dict.")
                raise RuntimeError()

        lora_params = (
            "w_a",
            "w_b",
            "obj_score_token",  # Parameters not present in SAM mask decoder
            "pred_obj_score_head",  # Parameters not present in SAM mask decoder
        )

        missing_keys, unexpected_keys = self.net.load_state_dict(
            state_dict, strict=False
        )

        missing_keys = [
            k for k in missing_keys if not any([key in k for key in lora_params])
        ]

        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()

        if unexpected_keys:
            logger.error(unexpected_keys)
            raise RuntimeError()

        logger.info(
            "%s loaded checkpoint from %s successfully.",
            self.net.__class__.__name__,
            checkpoint_path,
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


# Mask decoder factory classes


@register_mask_decoder("sam_mask_decoder")
class SAMMaskDecoderFactory(MaskDecoderFactory):
    """
    Factory class SAM mask decoder.
    """

    def build(self, args) -> SAMMaskDecoder:
        return SAMMaskDecoder(
            SAMMaskDecoderImpl(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=args.prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=args.prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                pred_obj_scores=args.get("pred_obj_scores", True),
                pred_obj_scores_mlp=args.get("pred_obj_scores_mlp", False),
            )
        )


@register_mask_decoder("lora_mask_decoder")
class LoRAMaskDecoderFactory(MaskDecoderFactory):
    """
    Factory class for LoRA mask decoders.
    """

    def _apply_lora(self, attn_block, r):
        """Helper method to apply LoRA to attention blocks."""
        input_dim = attn_block.embedding_dim
        output_dim = attn_block.internal_dim

        q_proj = attn_block.q_proj
        v_proj = attn_block.v_proj

        w_a_q = nn.Linear(input_dim, r, bias=False)
        w_b_q = nn.Linear(r, output_dim, bias=False)
        w_a_v = nn.Linear(input_dim, r, bias=False)
        w_b_v = nn.Linear(r, output_dim, bias=False)

        # initialize parameters
        self.reset_parameters(w_a_q, w_b_q)
        self.reset_parameters(w_a_v, w_b_v)
        attn_block.q_proj = LoRAqkv(q_proj, w_a_q, w_b_q)
        attn_block.v_proj = LoRAqkv(v_proj, w_a_v, w_b_v)

    def reset_parameters(self, w_a, w_b) -> None:
        nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b.weight)

    def build(self, args) -> LoRAMaskDecoder:
        mask_decoder = SAMMaskDecoderImpl(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=args.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=args.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            pred_obj_scores=args.get("pred_obj_scores", True),
            pred_obj_scores_mlp=args.get("pred_obj_scores_mlp", True),
        )
        assert args.r > 0, "r must be a positive integer."

        for param in mask_decoder.transformer.parameters():
            param.requires_grad = False

        decoder_transformer = mask_decoder.transformer
        for _, blk in enumerate(decoder_transformer.layers):
            self._apply_lora(blk.self_attn, args.r)
            self._apply_lora(blk.cross_attn_token_to_image, args.r)
            self._apply_lora(blk.cross_attn_image_to_token, args.r)

        # Apply LoRA to the final attention token to image block
        final_block = decoder_transformer.final_attn_token_to_image
        self._apply_lora(final_block, args.r)

        return LoRAMaskDecoder(mask_decoder)


class SAM2MaskDecoder(MaskDecoder):
    """
    SAM mask decoder for full-finetuning.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = []

        for name, param in self.net.named_parameters():
            if any((_train in name) for _train in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def predict_masks(self, *args, **kwargs):
        return self.net.predict_masks(*args, **kwargs)

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        # Extract the mask decoder state dict from model checkpoint
        state_dict = {
            k.replace("sam_mask_decoder.", ""): v
            for k, v in state_dict.items()
            if "sam_mask_decoder" in k
        }
        self.net.load_state_dict(state_dict)

        logger.info(
            "%s loaded from checkpoint %s successfully.",
            self.net.__class__.__name__,
            checkpoint_path,
        )


def build_decoder_mlp(decoder_config):
    """
    Build heads for mask decoder.
    - Object prediction head
    - Object pointer projection
    """
    from sam2rad.models.sam2.modeling.sam2_utils import MLP

    if decoder_config.use_obj_ptrs_in_encoder:
        # a linear projection on SAM output tokens to turn them into object pointers
        obj_ptr_proj = torch.nn.Linear(
            decoder_config.hidden_dim, decoder_config.hidden_dim
        )
        if decoder_config.use_mlp_for_obj_ptr_proj:
            obj_ptr_proj = MLP(
                decoder_config.hidden_dim,
                decoder_config.hidden_dim,
                decoder_config.hidden_dim,
                3,
            )
    else:
        obj_ptr_proj = torch.nn.Identity()
    if decoder_config.proj_tpos_enc_in_obj_ptrs:
        # a linear projection on temporal positional encoding in object pointers to
        # avoid potential interference with spatial positional encoding
        obj_ptr_tpos_proj = torch.nn.Linear(
            decoder_config.hidden_dim, decoder_config.mem_dim
        )
    else:
        obj_ptr_tpos_proj = torch.nn.Identity()

    return obj_ptr_proj, obj_ptr_tpos_proj


@register_mask_decoder("sam2_mask_decoder")
class SAM2MaskDecoderFactory(MaskDecoderFactory):
    """
    Factory class SAM mask decoder.
    """

    def build(self, args) -> SAM2MaskDecoder:
        """Build SAM-style mask decoder."""
        from sam2rad.models.sam2.modeling.sam.mask_decoder import (
            MaskDecoder as SAM2MaskDecoderImpl,
        )
        from sam2rad.models.sam2.modeling.sam.transformer import TwoWayTransformer

        decoder_config = args

        sam_prompt_embed_dim = decoder_config.hidden_dim

        sam_mask_decoder = SAM2MaskDecoderImpl(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=decoder_config.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=decoder_config.iou_prediction_use_sigmoid,
            pred_obj_scores=decoder_config.pred_obj_scores,
            pred_obj_scores_mlp=decoder_config.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=decoder_config.use_multimask_token_for_obj_ptr,
            **(decoder_config.sam_mask_decoder_extra_args or {}),
        )

        return SAM2MaskDecoder(sam_mask_decoder)


class SAM2LoRAMaskDecoder(SAM2MaskDecoder):
    def freeze_pretrained_parameters(self):
        trainable_modules = [
            "q_proj.w_a.weight",
            "q_proj.w_b.weight",
            "v_proj.w_a.weight",
            "v_proj.w_b.weight",
            "pred_obj_score_head",
            "obj_score_token",
            "mask_tokens",
            "output_upscaling",
            "output_hypernetworks_mlps",
        ]

        for name, param in self.net.named_parameters():
            if any((_train in name) for _train in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        # Extract the mask decoder state dict from model checkpoint
        state_dict = {
            k.replace("sam_mask_decoder.", ""): v
            for k, v in state_dict.items()
            if "sam_mask_decoder" in k
        }
        new_state_dict = {}

        for k in state_dict:
            # remap keys '.....q_proj....' -> '.....q_proj.proj....'
            # '.....v_proj....' -> '.....v_proj.proj....'
            if "q_proj" in k:
                new_state_dict[k.replace("q_proj", "q_proj.proj")] = state_dict[k]

            elif "v_proj" in k:
                new_state_dict[k.replace("v_proj", "v_proj.proj")] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]

        state_dict = new_state_dict

        # Sanity check
        for k in state_dict:
            if k not in self.net.state_dict().keys():
                logger.error(f"Key {k} not found in model state_dict.")

        lora_params = ("w_a", "w_b")
        missing_keys, unexpected_keys = self.net.load_state_dict(
            state_dict, strict=False
        )
        missing_keys = {
            k for k in missing_keys if not any([key in k for key in lora_params])
        }

        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()

        if unexpected_keys:
            logger.error(unexpected_keys)
            raise RuntimeError()

        logger.info(
            "%s loaded checkpoint from %s successfully.",
            self.net.__class__.__name__,
            checkpoint_path,
        )


@register_mask_decoder("sam2_lora_mask_decoder")
class SAM2LoRAMaskDecoderFactory(MaskDecoderFactory):
    """
    Factory class for sam-like LoRA mask decoders.
    """

    def _apply_lora(self, attn_block, r):
        """Helper method to apply LoRA to attention blocks."""
        input_dim = attn_block.embedding_dim
        output_dim = attn_block.internal_dim

        q_proj = attn_block.q_proj
        v_proj = attn_block.v_proj

        w_a_q = nn.Linear(input_dim, r, bias=False)
        w_b_q = nn.Linear(r, output_dim, bias=False)
        w_a_v = nn.Linear(input_dim, r, bias=False)
        w_b_v = nn.Linear(r, output_dim, bias=False)

        # initialize parameters
        self.reset_parameters(w_a_q, w_b_q)
        self.reset_parameters(w_a_v, w_b_v)
        attn_block.q_proj = LoRAqkv(q_proj, w_a_q, w_b_q)
        attn_block.v_proj = LoRAqkv(v_proj, w_a_v, w_b_v)

    def reset_parameters(self, w_a, w_b) -> None:
        nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b.weight)

    def build(self, args) -> SAM2LoRAMaskDecoder:
        """Build SAM-style LoRA mask decoder."""
        from sam2rad.models.sam2.modeling.sam.mask_decoder import (
            MaskDecoder as SAM2MaskDecoderImpl,
        )
        from sam2rad.models.sam2.modeling.sam.transformer import TwoWayTransformer

        decoder_config = args

        sam_prompt_embed_dim = decoder_config.hidden_dim

        sam_mask_decoder = SAM2MaskDecoderImpl(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=decoder_config.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=decoder_config.iou_prediction_use_sigmoid,
            pred_obj_scores=decoder_config.pred_obj_scores,
            pred_obj_scores_mlp=decoder_config.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=decoder_config.use_multimask_token_for_obj_ptr,
            **(decoder_config.sam_mask_decoder_extra_args or {}),
        )

        assert args.r > 0, "r must be a positive integer."

        for param in sam_mask_decoder.parameters():
            param.requires_grad = False

        decoder_transformer = sam_mask_decoder.transformer
        for _, blk in enumerate(decoder_transformer.layers):
            self._apply_lora(blk.self_attn, args.r)
            self._apply_lora(blk.cross_attn_token_to_image, args.r)
            self._apply_lora(blk.cross_attn_image_to_token, args.r)

        # Apply LoRA to the final attention token to image block
        final_block = decoder_transformer.final_attn_token_to_image
        self._apply_lora(final_block, args.r)

        return SAM2LoRAMaskDecoder(sam_mask_decoder)
