import logging
from abc import abstractmethod
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from  .MedMamba import VSSLayer
from sam2rad.models.adapters.adapters import AdapterBlock, adapter_residual
from sam2rad.models.sam.modeling import ImageEncoderViT
from sam2rad.models.sam.modeling.tiny_vit_sam import TinyViT

from .base import ImageEncoder, ImageEncoderFactory
from .registry import register_image_encoder


logger = logging.getLogger("sam2rad")


def resize_pos_embedding(sam_state_dict, new_state_dict, image_size, vit_patch_size):
    """
    Resize positional embedding match new image size.
    """

    ignore_keys = ["pos_embed", "rel_pos"]
    pos_embed = sam_state_dict["pos_embed"]
    token_size = int(image_size // vit_patch_size)

    if pos_embed.shape[1] != token_size:
        # Copy pre-trained state dict to new state dict
        new_state_dict.update(
            {
                k: v
                for k, v in sam_state_dict.items()
                if all(
                    ignore not in k for ignore in ignore_keys
                )  # Do not copy state dict for pos embeds
            }
        )
        # Interpolate positional embedding to match the new image size
        # resize pos embedding, which may sacrifice the performance
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(
            pos_embed, (token_size, token_size), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict["pos_embed"] = pos_embed
        rel_pos_keys = [k for k in sam_state_dict.keys() if "rel_pos" in k]

        global_rel_pos_keys = [
            k
            for k in rel_pos_keys
            if "2" in k
            or "5" in k
            or "7" in k
            or "8" in k
            or "11" in k
            or "13" in k
            or "15" in k
            or "23" in k
            or "31" in k
        ]

        for k in global_rel_pos_keys:
            h_check, w_check = sam_state_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(
                    rel_pos_params, (h, w), mode="bilinear", align_corners=False
                )

            new_state_dict[k] = rel_pos_params[0, 0, ...]

        return new_state_dict

    return sam_state_dict


class ViTBackbone(ImageEncoderViT):
    """
    SAM ImageEncoder with adapter layers.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        prompt_embed_dim,
        *args,
        **kwargs,
    ):
        super().__init__(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        self.freeze_pretrained_parameters()
        self.mamba_layers = VSSLayer(
            dim=768,
            depth=12,
            drop_path=0.33,
            attn_drop=0.1
        )
        self.combiner = GatedCombiner(768)

    def freeze_pretrained_parameters(self):
        # 解冻 mamba_layers 中的参数
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        high_res_features = []
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        mambax, reduced_outputs = self.mamba_layers(x)
        for i, blk in enumerate(self.blocks):
            x, blk_mambax = blk(x,reduced_outputs[i])
            fusionx = x + blk_mambax
            high_res_features.append(fusionx.permute(0, 3, 1, 2))

        x = self.combiner(x,mambax)
        x = self.neck(x.permute(0, 3, 1, 2))


        return x, high_res_features[-2:]


class GatedCombiner(nn.Module):
    def __init__(self, input_dim):
        super(GatedCombiner, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, mambax):
        gate = self.gate(x)
        x = gate * x + (1 - gate) * mambax
        return x

        return combined




# Image encoders with adapter layers
class SAMAdapter(ImageEncoder):
    """
    ViT image encoder with adapters.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.img_size = self.net.img_size
        self.mamba_layers = self.net.mamba_layers
        self.combiner = self.net.combiner

        self.adapters = nn.ModuleList()
        for block in self.net.blocks[-5:]:
            adapter = AdapterBlock(block)
            self.adapters.append(adapter)
            block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = ["adapters", "patch_adapter", "lora","mamba_layers","net.mamba_layers","combiner","net.combiner"]
        for name, param in self.net.named_parameters():
            if any(mod in name for mod in trainable_modules):
                param.requires_grad = True
                print(f"Trainable: {name}")
            else:
                param.requires_grad = False
        for name, param in self.mamba_layers.named_parameters():
            param.requires_grad = True
            print(f"Trainable: {name} (from mamba_layers)")
        for name, param in self.combiner.named_parameters():
            param.requires_grad = True
            print(f"Trainable: {name} (from combiner)")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load state dict from checkpoint.
        """
        sam_state_dict = torch.load(checkpoint_path)
        # Extract the image encoder state dict from model checkpoint
        sam_state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in sam_state_dict.items()
            if "image_encoder." in k
        }

        # Ignore keys for the adapter layers
        ignore_keys = ["adapters","mamba_layers","combiner"]
        # Resize positional embedding
        image_size, vit_patch_size = (
            self.img_size,
            16,
        )  # TODO: Fix patch size, should be from config

        state_dict = resize_pos_embedding(
            sam_state_dict, self.net.state_dict(), image_size, vit_patch_size
        )

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

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class SAMViTBAdapter(SAMAdapter):
    """
    ViT-B image encoder with adapters.
    """


class SAMViTLAdapter(SAMAdapter):
    """
    ViT-L image encoder with adapters.
    """


class SAMViTHAdapter(SAMAdapter):
    """
    ViT-H image encoder with adapters.
    """


# Image encoder without adapter layers
class SAMImageEncoder(ImageEncoder):
    """
    Default SAM image encoder base class.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.img_size = self.net.img_size

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load state dict from checkpoint.
        """
        sam_state_dict = torch.load(checkpoint_path)
        # Extract the image encoder state dict from model checkpoint
        sam_state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in sam_state_dict.items()
            if "image_encoder." in k
        }

        state_dict = resize_pos_embedding(
            sam_state_dict,
            self.net.state_dict(),
            self.img_size,
            16,  # TODO: Fix patch size, should be from config
        )

        self.net.load_state_dict(state_dict)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class SAMViTBImageEncoder(SAMImageEncoder):
    """
    SAM image encoder with ViT-B architecture.
    """


class SAMViTLImageEncoder(SAMImageEncoder):
    """
    SAM image encoder with ViT-L architecture.
    """


class SAMViTHImageEncoder(SAMImageEncoder):
    """
    SAM image encoder with ViT-H architecture.
    """


# Image encoder factory classes with adapter layers
class SAMAdapterFactory(ImageEncoderFactory):
    """
    SAM image encoder with adapter layers.
    """

    @abstractmethod
    def build(self, args) -> SAMAdapter:
        pass


@register_image_encoder("sam_vit_b_adapter")
class SAMViTBAdapterFactory(SAMAdapterFactory):
    """
    ViT-B image encoder with adapters.
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
        )

    def build(self, args) -> SAMViTBAdapter:
        backbone = self.get_backbone(args)
        return SAMViTBAdapter(backbone)


@register_image_encoder("sam_vit_l_adapter")
class SAMViTLAdapterFactory(ImageEncoderFactory):
    """
    Sam image encoder ViT-L with adapters.
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[5, 11, 17, 23],
        )

    def build(self, args) -> SAMViTBAdapter:
        backbone = self.get_backbone(args)
        return SAMViTBAdapter(backbone)


@register_image_encoder("sam_vit_h_adapter")
class SAMViTHAdapterFactory(ImageEncoderFactory):
    """
    Sam image encoder with ViT-H.
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
        )

    def build(self, args) -> SAMViTBAdapter:
        backbone = self.get_backbone(args)
        return SAMViTBAdapter(backbone)


# Image encoder factory classes without adapter layers
class SAMImageEncoderFactory(ImageEncoderFactory):
    """
    Base class for default SAM image encoders.
    """

    @abstractmethod
    def build(self, args) -> SAMImageEncoder:
        pass


@register_image_encoder("sam_vit_b")
class SAMViTBFactory(SAMImageEncoderFactory):
    """
    Concrete image encoder factories
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
        )

    def build(self, args) -> ViTBackbone:
        return self.get_backbone(args)


@register_image_encoder("sam_vit_l")
class SAMViTLFactory(SAMImageEncoderFactory):
    """
    Sam image encoder with ViT-L.
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[5, 11, 17, 23],
        )

    def build(self, args) -> ViTBackbone:
        return self.get_backbone(args)


@register_image_encoder("sam_vit_h")
class SAMViTHFactory(SAMImageEncoderFactory):
    """
    Sam image encoder with ViT-H.
    """

    def get_backbone(self, args) -> nn.Module:
        return ViTBackbone(
            image_size=args.image_size,
            patch_size=args.patch_size,
            prompt_embed_dim=args.prompt_embed_dim,
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
        )

    def build(self, args) -> ViTBackbone:
        return self.get_backbone(args)


class TinyViTImageEncoder(ImageEncoder):
    """
    Image encoder with tiny ViT architecture.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.img_size = self.net.img_size

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = ["adapters", "patch_adapter", "lora","mamba_layers","trunk.mamba_layers"]
        for name, param in self.net.named_parameters():
            if any(mod in name for mod in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load state dict from checkpoint.
        """
        state_dict = torch.load(checkpoint_path)
        # Extract the image encoder state dict from model checkpoint
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if "image_encoder." in k
        }

        # Ignore keys for adapter layers
        ignore_keys = ["adapters", "patch_adapter"]
        except_keys = ["norm_head", "head"]
        missing_keys, unexpected_keys = self.net.load_state_dict(
            state_dict, strict=False
        )

        missing_keys = {
            k for k in missing_keys if not any([key in k for key in ignore_keys])
        }

        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()

        unexpected_keys = {
            k for k in unexpected_keys if not any([key in k for key in except_keys])
        }

        if unexpected_keys:
            logger.erro(unexpected_keys)
            raise RuntimeError()

        logger.info(
            "{} loaded checkpoint from {} successfully.".format(
                self.net.__class__.__name__,
                checkpoint_path,
            )
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


@register_image_encoder("vit_tiny")
class TinyViTFactory(ImageEncoderFactory):
    """
    Image encoder with tiny ViT architecture.
    """

    def build(self, args) -> TinyViTImageEncoder:
        backbone = self.get_backbone(args)
        return TinyViTImageEncoder(backbone)

    def get_backbone(self, args) -> nn.Module:
        return TinyViT(
            img_size=args.image_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )


class TinyViTAdapter(TinyViTImageEncoder):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

        self.adapters = nn.ModuleList()
        for layer in self.net.layers[1:]:
            for block in layer.blocks:
                adapter = AdapterBlock(block)
                self.adapters.append(adapter)
                block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()


@register_image_encoder("vit_tiny_adapter")
class TinyViTAdapterFactory(TinyViTFactory):
    def build(self, args) -> TinyViTAdapter:
        backbone = self.get_backbone(args)
        return TinyViTAdapter(backbone)


# SAM2: https://arxiv.org/abs/2408.00714
class Sam2TinyHiera(ImageEncoder):
    """
    SAM2 with tiny image encoder.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        trainable_modules = ["adapters", "patch_adapter", "lora","mamba_layers","trunk.mamba_layers"]
        for name, param in self.net.named_parameters():
            if any(mod in name for mod in trainable_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load state dict from checkpoint.
        """
        # print("Loading checkpoint from11111111111111111111111111111111111111111111111", checkpoint_path)
        state_dict = torch.load(checkpoint_path)["model"]
        # Extract the image encoder state dict from model checkpoint
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if "image_encoder." in k
        }

        # Ignore keys for adapter layers
        ignore_keys = ["adapters", "patch_adapter","mamba_layers","trunk.mamba_layers"]
        except_keys = []
        missing_keys, unexpected_keys = self.net.load_state_dict(
            state_dict, strict=False
        )

        missing_keys = {
            k for k in missing_keys if not any([key in k for key in ignore_keys])
        }

        if missing_keys:
            logger.error(missing_keys)
            raise RuntimeError()

        unexpected_keys = {
            k for k in unexpected_keys if not any([key in k for key in except_keys])
        }

        if unexpected_keys:
            logger.erro(unexpected_keys)
            raise RuntimeError()

        logger.info(
            "{} loaded checkpoint from {} successfully.".format(
                self.net.__class__.__name__,
                checkpoint_path,
            )
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


@register_image_encoder("sam2_tiny_hiera")
class Sam2TinyHieraFactory(ImageEncoderFactory):
    """
    SAM2 image encoder factory.
    """

    def build(self, *args, **kwargs) -> Sam2TinyHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2TinyHiera(backbone)

    def get_backbone(self, *args, **kwargs) -> nn.Module:
        from sam2rad.models.sam2.modeling.backbones.hieradet import Hiera
        from sam2rad.models.sam2.modeling.backbones.image_encoder import (
            FpnNeck,
            ImageEncoder,
        )
        from sam2rad.models.sam2.modeling.position_encoding import PositionEmbeddingSine

        # configs
        scalp = 1
        trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[7, 7],
        )

        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        backbone = ImageEncoder(
            scalp=scalp,
            trunk=trunk,
            neck=neck,
        )

        return backbone


class Sam2TinyHieraAdapter(Sam2TinyHiera):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

        self.adapters = nn.ModuleList()
        for block in self.net.trunk.blocks:
            adapter = AdapterBlock(block)
            self.adapters.append(adapter)
            block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()


@register_image_encoder("sam2_tiny_hiera_adapter")
class Sam2TinyHieraAdapterFactory(Sam2TinyHieraFactory):
    """
    SAM2 tiny image encoder with adapters.
    """

    def build(self, *args, **kwargs) -> Sam2TinyHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2TinyHieraAdapter(backbone)


class Sam2SmallHiera(Sam2TinyHiera):
    """
    SAM2 with small image encoder.
    """


class Sam2SmallHieraAdapter(Sam2SmallHiera):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

        self.adapters = nn.ModuleList()
        for block in self.net.trunk.blocks:
            adapter = AdapterBlock(block)
            self.adapters.append(adapter)
            block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()


@register_image_encoder("sam2_small_hiera")
class Sam2SmallHieraFactory(ImageEncoderFactory):
    """
    SAM2 image encoder factory.
    """

    def build(self, *args, **kwargs) -> Sam2SmallHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2SmallHiera(backbone)

    def get_backbone(self, *args, **kwargs) -> nn.Module:
        from sam2rad.models.sam2.modeling.backbones.hieradet import Hiera
        from sam2rad.models.sam2.modeling.backbones.image_encoder import (
            FpnNeck,
            ImageEncoder,
        )
        from sam2rad.models.sam2.modeling.position_encoding import PositionEmbeddingSine

        # configs
        scalp = 1
        trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 11, 2],
            global_att_blocks=[7, 10, 13],
            window_pos_embed_bkg_spatial_size=[7, 7],
        )

        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        backbone = ImageEncoder(
            scalp=scalp,
            trunk=trunk,
            neck=neck,
        )

        return backbone


@register_image_encoder("sam2_small_hiera_adapter")
class Sam2SmallHieraAdapterFactory(Sam2SmallHieraFactory):
    def build(self, *args, **kwargs) -> Sam2SmallHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2SmallHieraAdapter(backbone)


class Sam2LargeHiera(Sam2TinyHiera):
    """
    SAM2 with large image encoder.
    """


@register_image_encoder("sam2_large_hiera")
class Sam2LargeHieraFactory(ImageEncoderFactory):
    """
    SAM2 large image encoder factory.
    """

    def build(self, *args, **kwargs) -> Sam2LargeHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2LargeHiera(backbone)

    def get_backbone(self, *args, **kwargs) -> nn.Module:
        from sam2rad.models.sam2.modeling.backbones.hieradet import Hiera
        from sam2rad.models.sam2.modeling.backbones.image_encoder import (
            FpnNeck,
            ImageEncoder,
        )
        from sam2rad.models.sam2.modeling.position_encoding import PositionEmbeddingSine

        # configs
        scalp = 1
        trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        )

        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        backbone = ImageEncoder(
            scalp=scalp,
            trunk=trunk,
            neck=neck,
        )

        return backbone


class Sam2LargeHieraAdapter(Sam2LargeHiera):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

        self.adapters = nn.ModuleList()
        for block in self.net.trunk.blocks:
            adapter = AdapterBlock(block)
            self.adapters.append(adapter)
            block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()


@register_image_encoder("sam2_large_hiera_adapter")
class Sam2LargeHieraAdapterFactory(Sam2LargeHieraFactory):
    def build(self, *args, **kwargs) -> Sam2LargeHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2LargeHieraAdapter(backbone)


class Sam2BasePlusHiera(Sam2TinyHiera):
    """
    SAM2 with Big image encoder.
    """


class Sam2BasePlusHieraAdapter(Sam2BasePlusHiera):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

        self.adapters = nn.ModuleList()
        for block in self.net.trunk.blocks:
            adapter = AdapterBlock(block)
            self.adapters.append(adapter)
            block.register_forward_pre_hook(partial(adapter_residual, adapter))

        self.freeze_pretrained_parameters()


@register_image_encoder("sam2_base+_hiera")
class Sam2BasePlusHieraFactory(ImageEncoderFactory):
    """
    SAM2 big image encoder factory.
    """

    def build(slef, *args, **kwargs) -> Sam2BasePlusHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2BasePlusHiera(backbone)

    def get_backbone(self, *args, **kwargs) -> nn.Module:
        from sam2rad.models.sam2.modeling.backbones.hieradet import Hiera
        from sam2rad.models.sam2.modeling.backbones.image_encoder import (
            FpnNeck,
            ImageEncoder,
        )
        from sam2rad.models.sam2.modeling.position_encoding import PositionEmbeddingSine

        # configs
        scalp = 1
        trunk = Hiera(
            embed_dim=112,
            num_heads=2,
        )

        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        backbone = ImageEncoder(
            scalp=scalp,
            trunk=trunk,
            neck=neck,
        )

        return backbone


@register_image_encoder("sam2_base+_hiera_adapter")
class Sam2BasePlusHieraAdapterFactory(Sam2BasePlusHieraFactory):
    def build(self, *args, **kwargs) -> Sam2BasePlusHiera:
        backbone = self.get_backbone(*args, **kwargs)
        return Sam2BasePlusHieraAdapter(backbone)
