import logging

import torch
import yaml
from sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)
from sam2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)

from sam2rad.blob.misc import DotDict
from sam2rad.decoders.build_decoder import build_decoder_mlp
from sam2rad.decoders.registry import MASK_DECODER_REGISTRY
from sam2rad.encoders.registry import IMAGE_ENCODER_REGISTRY
from sam2rad.models.ppn import PROMPT_PREDICTORS, PromptSampler
from sam2rad.models.sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2rad.models.sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2rad.models.sam2.modeling.sam.transformer import RoPEAttention

from .model import Model

logger = logging.getLogger("sam2rad")

# fmt: off
CONFIG = """
# Model
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  image_encoder:
    _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
    scalp: 1
    trunk:
      _target_: sam2.modeling.backbones.hieradet.Hiera
      embed_dim: 96
      num_heads: 1
      stages: [1, 2, 7, 2]
      global_att_blocks: [5, 7, 9]
      window_pos_embed_bkg_spatial_size: [7, 7]
    neck:
      _target_: sam2.modeling.backbones.image_encoder.FpnNeck
      position_encoding:
        _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
        num_pos_feats: 256
        normalize: true
        scale: null
        temperature: 10000
      d_model: 256
      backbone_channel_list: [768, 384, 192, 96]
      fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features
      fpn_interp_model: nearest

  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    pos_enc_at_input: true
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
      dim_feedforward: 2048
      dropout: 0.1
      pos_enc_at_attn: false
      self_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttention
        rope_theta: 10000.0
        feat_sizes: [32, 32]
        embedding_dim: 256
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
      d_model: 256
      pos_enc_at_cross_attn_keys: true
      pos_enc_at_cross_attn_queries: false
      cross_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttention
        rope_theta: 10000.0
        feat_sizes: [32, 32]
        rope_k_repeat: True
        embedding_dim: 256
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
        kv_in_dim: 64
    num_layers: 4

  memory_encoder:
      _target_: sam2.modeling.memory_encoder.MemoryEncoder
      out_dim: 64
      position_encoding:
        _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
        num_pos_feats: 64
        normalize: true
        scale: null
        temperature: 10000
      mask_downsampler:
        _target_: sam2.modeling.memory_encoder.MaskDownSampler
        kernel_size: 3
        stride: 2
        padding: 1
      fuser:
        _target_: sam2.modeling.memory_encoder.Fuser
        layer:
          _target_: sam2.modeling.memory_encoder.CXBlock
          dim: 256
          kernel_size: 7
          padding: 3
          layer_scale_init_value: 1.0e-6
          use_dwconv: True  # depth-wise convs
        num_layers: 2

  num_maskmem: 7
  image_size: 1024
  # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
  # SAM decoder
  sigmoid_scale_for_mem_enc: 20.0
  sigmoid_bias_for_mem_enc: -10.0
  use_mask_input_as_output_without_sam: true
  # Memory
  directly_add_no_mem_embed: true
  # use high-resolution feature map in the SAM mask decoder
  use_high_res_features_in_sam: true
  # output 3 masks on the first click on initial conditioning frames
  multimask_output_in_sam: true
  # SAM heads
  iou_prediction_use_sigmoid: True
  # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
  use_obj_ptrs_in_encoder: true
  add_tpos_enc_to_obj_ptrs: true
  only_obj_ptrs_in_the_past_for_eval: true
  # object occlusion prediction
  pred_obj_scores: true
  pred_obj_scores_mlp: true
  fixed_no_obj_ptr: true
  # multimask tracking settings
  multimask_output_for_tracking: true
  use_multimask_token_for_obj_ptr: true
  multimask_min_pt_num: 0
  multimask_max_pt_num: 1
  use_mlp_for_obj_ptr_proj: true
  # Compilation flag
  # HieraT does not currently support compilation, should always be set to False
  compile_image_encoder: False
  backbone_stride: 16
  binarize_mask_from_pts_for_mem_enc: false
  max_cond_frames_in_attn: -1
  memory_temporal_stride_for_eval: 1
  add_all_frames_to_correct_as_cond: false
  non_overlap_masks_for_mem_enc: false
  max_obj_ptrs_in_encoder: 16
  proj_tpos_enc_in_obj_ptrs: true
  soft_no_obj_ptr: false
  sam_mask_decoder_extra_args: null
"""

# fmt: on
CONFIG = yaml.safe_load(CONFIG)
CONFIG = DotDict.from_dict(CONFIG)


def build_model(args) -> Model:
    """
    Build model by choosing from a range of image encoder and decoders.
    """

    # Build encoder
    image_encoder = IMAGE_ENCODER_REGISTRY[args.image_encoder]().build(args)

    try:
        image_encoder.load_checkpoint(args.sam_checkpoint)
    except RuntimeError:
        # We only need to load pre-trained SAM checkpoint during training
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise

    # TODO: get the parameters from the config file
    # Build memory attention
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=RoPEAttention(
                rope_theta=CONFIG.model.memory_attention.layer.self_attention.rope_theta,
                feat_sizes=CONFIG.model.memory_attention.layer.self_attention.feat_sizes,
                embedding_dim=CONFIG.model.memory_attention.layer.self_attention.embedding_dim,
                num_heads=CONFIG.model.memory_attention.layer.self_attention.num_heads,
                downsample_rate=CONFIG.model.memory_attention.layer.self_attention.downsample_rate,
                dropout=CONFIG.model.memory_attention.layer.self_attention.dropout,
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            ),
        ),
        num_layers=4,
    )

    for param in memory_attention.parameters():
        param.requires_grad = False

    # Load checkpoint
    try:
        state_dict = torch.load(args.sam_checkpoint)["model"]
        # Extract the memory attention state dict from model checkpoint
        state_dict = {
            k.replace("memory_attention.", ""): v
            for k, v in state_dict.items()
            if "memory_attention" in k
        }
        memory_attention.load_state_dict(state_dict)

    except RuntimeError:
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise

    logger.info(
        "%s loaded from checkpoint %s successfully.",
        memory_attention.__class__.__name__,
        args.sam_checkpoint,
    )

    # Build memory encoder
    memory_encoder = MemoryEncoder(
        out_dim=CONFIG.model.memory_encoder.out_dim,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=CONFIG.model.memory_encoder.position_encoding.num_pos_feats,
            normalize=CONFIG.model.memory_encoder.position_encoding.normalize,
            scale=CONFIG.model.memory_encoder.position_encoding.scale,
            temperature=CONFIG.model.memory_encoder.position_encoding.temperature,
        ),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(
                dim=CONFIG.model.memory_encoder.fuser.layer.dim,
                kernel_size=CONFIG.model.memory_encoder.fuser.layer.kernel_size,
                padding=CONFIG.model.memory_encoder.fuser.layer.padding,
                layer_scale_init_value=CONFIG.model.memory_encoder.fuser.layer.layer_scale_init_value,
                use_dwconv=CONFIG.model.memory_encoder.fuser.layer.use_dwconv,
            ),
            num_layers=CONFIG.model.memory_encoder.fuser.num_layers,
        ),
    )

    for param in memory_encoder.parameters():
        param.requires_grad = False

    # Load checkpoint
    try:
        state_dict = torch.load(args.sam_checkpoint)["model"]
        # Extract the memory encoder state dict from model checkpoint
        state_dict = {
            k.replace("memory_encoder.", ""): v
            for k, v in state_dict.items()
            if "memory_encoder" in k
        }
        memory_encoder.load_state_dict(state_dict)
    except RuntimeError:
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise

    logger.info(
        "%s loaded from checkpoint %s successfully.",
        memory_encoder.__class__.__name__,
        args.sam_checkpoint,
    )

    # Build prompt encoder
    args.image_size = args.dataset.image_size
    sam_image_embedding_size = args.image_size // CONFIG.model.backbone_stride
    prompt_encoder = PromptEncoder(
        embed_dim=memory_attention.d_model,
        image_embedding_size=(
            sam_image_embedding_size,
            sam_image_embedding_size,
        ),
        input_image_size=(args.image_size, args.image_size),
        mask_in_chans=16,
    )

    try:
        state_dict = torch.load(args.sam_checkpoint)["model"]
        # Extract the prompt encoder state dict from model checkpoint
        state_dict = {
            k.replace("sam_prompt_encoder.", ""): v
            for k, v in state_dict.items()
            if "sam_prompt_encoder" in k
        }

        prompt_encoder.load_state_dict(state_dict)
    except RuntimeError:
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise
    logger.info(
        "%s loaded from checkpoint %s successfully.",
        prompt_encoder.__class__.__name__,
        args.sam_checkpoint,
    )

    prompt_sampler = PromptSampler(
        prompt_learner=PROMPT_PREDICTORS[
            args.get("prompt_predictor", "sam2_high_res_ppn")
        ](
            prompt_encoder=prompt_encoder,
            embedding_dim=prompt_encoder.embed_dim,
            num_heads=8,
            mlp_dim=256 * 8,
        ),
        prompt_encoder=prompt_encoder,
    )

    logger.info("Prompt sampler loaded successfully.")

    # Build decoder
    CONFIG.model.update(
        {
            "hidden_dim": memory_attention.d_model,
            "mem_dim": image_encoder.net.neck.d_model,
            "r": 8,  # if LoRA is used
        }
    )
    mask_decoder = MASK_DECODER_REGISTRY[args.mask_decoder]().build(CONFIG.model)
    try:
        mask_decoder.load_checkpoint(args.sam_checkpoint)
    except RuntimeError:
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise

    # Build decoder MLPs
    obj_ptr_proj, obj_ptr_tpos_proj = build_decoder_mlp(CONFIG.model)

    # load checkpoint
    try:
        state_dict = torch.load(args.sam_checkpoint)["model"]
        # Extrap MLP parameters from model checkpoint
        state_dict_obj_ptr_proj = {
            k.replace("obj_ptr_proj.", ""): v
            for k, v in state_dict.items()
            if "obj_ptr_proj" in k
        }

        state_dict_obj_ptr_tpos_proj = {
            k.replace("obj_ptr_tpos_proj.", ""): v
            for k, v in state_dict.items()
            if "obj_ptr_tpos_proj" in k
        }
        # obj_ptr_proj.load_state_dict(state_dict_obj_ptr_proj)
        # obj_ptr_tpos_proj.load_state_dict(state_dict_obj_ptr_tpos_proj)
    except RuntimeError:
        logger.error("No SAM checkpoint loaded. Loading without pre-trained weights.")
        raise

    logger.info(
        "%s and %s loaded from checkpoint %s successfully.",
        obj_ptr_proj.__class__.__name__,
        obj_ptr_tpos_proj.__class__.__name__,
        args.sam_checkpoint,
    )

    model = Model(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_sampler=prompt_sampler,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        obj_ptr_proj=obj_ptr_proj,
        obj_ptr_tpos_proj=obj_ptr_tpos_proj,
        num_maskmem=CONFIG.model.num_maskmem,
        image_size=args.image_size,
        backbone_stride=CONFIG.model.backbone_stride,
        sigmoid_scale_for_mem_enc=CONFIG.model.sigmoid_scale_for_mem_enc,
        sigmoid_bias_for_mem_enc=CONFIG.model.sigmoid_bias_for_mem_enc,
        binarize_mask_from_pts_for_mem_enc=CONFIG.model.binarize_mask_from_pts_for_mem_enc,
        use_mask_input_as_output_without_sam=CONFIG.model.use_mask_input_as_output_without_sam,
        max_cond_frames_in_attn=CONFIG.model.max_cond_frames_in_attn,
        directly_add_no_mem_embed=CONFIG.model.directly_add_no_mem_embed,
        use_high_res_features_in_sam=CONFIG.model.use_high_res_features_in_sam,
        multimask_output_in_sam=CONFIG.model.multimask_output_in_sam,
        multimask_max_pt_num=CONFIG.model.multimask_max_pt_num,
        multimask_output_for_tracking=CONFIG.model.multimask_output_for_tracking,
        use_multimask_token_for_obj_ptr=CONFIG.model.use_multimask_token_for_obj_ptr,
        iou_prediction_use_sigmoid=CONFIG.model.iou_prediction_use_sigmoid,
        memory_temporal_stride_for_eval=CONFIG.model.memory_temporal_stride_for_eval,
        add_all_frames_to_correct_as_cond=CONFIG.model.add_all_frames_to_correct_as_cond,
        non_overlap_masks_for_mem_enc=CONFIG.model.non_overlap_masks_for_mem_enc,
        use_obj_ptrs_in_encoder=CONFIG.model.use_obj_ptrs_in_encoder,
        max_obj_ptrs_in_encoder=CONFIG.model.max_obj_ptrs_in_encoder,
        add_tpos_enc_to_obj_ptrs=CONFIG.model.add_tpos_enc_to_obj_ptrs,
        proj_tpos_enc_in_obj_ptrs=CONFIG.model.proj_tpos_enc_in_obj_ptrs,
        only_obj_ptrs_in_the_past_for_eval=CONFIG.model.only_obj_ptrs_in_the_past_for_eval,
        pred_obj_scores=CONFIG.model.pred_obj_scores,
        pred_obj_scores_mlp=CONFIG.model.pred_obj_scores_mlp,
        fixed_no_obj_ptr=CONFIG.model.fixed_no_obj_ptr,
        soft_no_obj_ptr=CONFIG.model.soft_no_obj_ptr,
        use_mlp_for_obj_ptr_proj=CONFIG.model.use_mlp_for_obj_ptr_proj,
        compile_image_encoder=CONFIG.model.compile_image_encoder,
    )

    # Load checkpoint for other model components
    state_dict = torch.load(args.sam_checkpoint)["model"]
    ignore_keys = [
        "image_encoder.",
        "sam_mask_decoder.",
        "sam_prompt_encoder.",
        "prompt_sampler.",
        "box_regression_head",
        "no_obj_embed_spatial",
        "obj_ptr_tpos_proj",
    ]

    state_dict = {
        k: v
        for k, v in state_dict.items()
        if all([key not in k for key in ignore_keys])
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )  # Caution: all modules are loaded correctly. strict=False to ensure remaining parameters are loaded correctly (e.g., model.no_obj_ptr)

    missing_keys = {
        k for k in missing_keys if not any([key in k for key in ignore_keys])
    }
    if missing_keys:
        logger.error("Missing keys: %r", missing_keys)

    if unexpected_keys:
        logger.error(unexpected_keys)
        raise RuntimeError()

    return model
