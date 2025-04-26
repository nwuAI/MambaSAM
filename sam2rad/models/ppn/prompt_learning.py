from typing import List, Type
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from sam2.modeling.sam2_utils import MLP

from .registry import register_prompt_predictor


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionEmbeddingSine1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionEmbeddingSine1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class BoxRegressionHead(torch.nn.Module):
    """
    Given box embeddings, predict the coordinates of the bounding box.
    """

    def __init__(self, *, in_features, hidden_dim, num_layers):
        super().__init__()
        self.mlp = MLP(in_features, hidden_dim, 4, num_layers)

    def forward(self, x):
        """
        x.shape: (B, 2, 256)
        """
        x = x.flatten(1)
        xywh = self.mlp(x)
        xywh = torch.sigmoid(xywh)
        x1y1 = xywh[:, :2]
        wh = xywh[:, 2:]
        xyxy = torch.cat([x1y1, x1y1 + wh], dim=1)
        return xyxy


class MaskClassifier(nn.Module):
    """
    Given mask embeddings, predict the binary mask.
    The network consists of multiple convolutional layers followed by upsampling
    to produce the final segmentation mask.

    Args:
        in_features (int): Number of input channels
        hidden_dim (int): Number of channels in hidden layers
        num_layers (int): Number of convolutional layers
        scale_factor (int): Upsampling factor for the final output
        dropout_rate (float): Dropout probability for regularization
        use_batch_norm (bool): Whether to use batch normalization
    """

    def __init__(
        self,
        *,
        in_features,
        hidden_dim,
        num_layers=2,
        scale_factor=4,
        dropout_rate=0.1,
        use_batch_norm=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        self.scale_factor = scale_factor

        # Build the convolutional layers
        curr_features = in_features
        for _ in range(num_layers):
            self.layers.append(
                nn.Conv2d(
                    curr_features,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not use_batch_norm,  # Disable bias when using batch norm
                )
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(hidden_dim))
            curr_features = hidden_dim

        # Final 1x1 conv to reduce to 1 channel for binary mask
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mask classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features, height, width)

        Returns:
            torch.Tensor: Predicted binary mask of shape (batch_size, 1, height*scale_factor, width*scale_factor)
        """
        # Apply convolutional layers with activation and optional batch norm
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Apply final 1x1 convolution
        x = self.final_conv(x)

        # Upsample to desired size
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        return x

    def init_weights(self):
        """Initialize the weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@register_prompt_predictor("linear")
class TwoWayCrossAttention(TwoWayAttentionBlock):
    def __init__(
        self,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__(
            embedding_dim,
            num_heads,
            mlp_dim,
            activation,
            attention_downsample_rate,
            skip_first_layer_pe,
        )
        self.pos_encoding = PositionEmbeddingSine(embedding_dim)
        self.pos_encoding_1d = PositionEmbeddingSine1D(embedding_dim)

        self.box_regression_head = BoxRegressionHead(
            in_features=256 * 2, hidden_dim=256, num_layers=2
        )

        self.mask_classifier = MaskClassifier(
            in_features=256, hidden_dim=64, scale_factor=4
        )
        self.prompt_encoder = prompt_encoder

        self.freeze_pretrained_weights()

    def freeze_pretrained_weights(self):
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        *,
        image_features: List[torch.Tensor],
        queries: torch.Tensor,
    ):
        # image_features: List of image features from the encoder (256 x 256, 128 x 128, 64 x 64)
        # Use the last layer of the image encoder
        image_embedding = image_features[-1]  # B x C x 64 x 64
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_pe = self.pos_encoding(image_embedding[:1])
        query_pe = self.pos_encoding_1d(queries)  # B x N_query_tokens x C
        image_embedding = image_embedding.flatten(2).permute(
            0, 2, 1
        )  # B x N_image_tokens x C
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # B x N_image_tokens x C

        sparse_embeddings, dense_embeddings = super().forward(
            queries=queries,
            keys=image_embedding,
            query_pe=query_pe,
            key_pe=image_pe,
        )
        dense_embeddings = dense_embeddings.view(bs, h, w, -1).permute(0, 3, 1, 2)

        # Predict bounding box and mask prompts
        interim_mask_output = self.mask_classifier(dense_embeddings)
        dense_embeddings = self.prompt_encoder._embed_masks(
            interim_mask_output
        )  # (B, 256, H//16, W//16)

        # Encode learned box coordinates
        pred_boxes = self.box_regression_head(sparse_embeddings[:, :2])
        learned_box_embeddings = self.prompt_encoder._embed_boxes(
            boxes=pred_boxes * self.prompt_encoder.input_image_size[0]
        )

        sparse_embeddings = torch.cat(
            [learned_box_embeddings, sparse_embeddings[:, 2:]], dim=1
        )

        return (
            sparse_embeddings,
            dense_embeddings,
            interim_mask_output,
            pred_boxes,
        )


class HighResPPN(nn.Module):
    def __init__(
        self,
        *,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        depth: int = 3,
        channel_dims: List[int] = [256, 64, 32],
        mask_scale_factor: int = 1,
    ) -> None:
        # depth: number of high resolution features from the image encoder
        super().__init__()
        self.pos_encoding2d = PositionEmbeddingSine(embedding_dim)
        self.pos_encoding1d = PositionEmbeddingSine1D(embedding_dim)

        self.box_regression_head = BoxRegressionHead(
            in_features=256 * 2, hidden_dim=256, num_layers=2
        )
        self.mask_classifier = MaskClassifier(
            in_features=256, hidden_dim=64, scale_factor=mask_scale_factor
        )
        self.prompt_encoder = prompt_encoder

        self.channel_adapters = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

            if channel_dims[i] == embedding_dim:
                self.channel_adapters.append(nn.Identity())
            else:
                self.channel_adapters.append(
                    nn.Conv2d(
                        channel_dims[i],
                        embedding_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

        self.freeze_pretrained_weights()

    def freeze_pretrained_weights(self):
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    @staticmethod
    def interp(x, size, mode="bilinear"):
        # x: (B, HW, C)
        # size: (H'W')
        bs, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.permute(0, 2, 1).reshape(bs, c, h, w)
        h_target = w_target = int(math.sqrt(size))
        x = F.interpolate(
            x,
            size=(h_target, w_target),
            mode=mode,
            align_corners=None if mode == "nearest" else False,
        )
        return x.flatten(2).permute(0, 2, 1)

    def forward(
        self,
        image_features: List[torch.Tensor],
        queries: torch.Tensor,
    ):
        image_features.reverse()  # High resolution to low resolution

        # Change the channel dimensions of high resolution features to match the embedding dimension
        image_features = [
            adapter(f) for f, adapter in zip(image_features, self.channel_adapters)
        ]
        query_pe = self.pos_encoding1d(queries)  # B x N_query_tokens x C
        bs, c, h, w = image_features[0].shape
        dense_embeddings = (
            self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            .expand(bs, -1, h, w)
            .flatten(2)
            .permute(0, 2, 1)
        )  # (B, H*W, c)

        for i, cross_attn_layer in enumerate(self.layers):
            # BxCxHxW -> BxHWxC == B x N_image_tokens x C
            # bs, c, h, w = image_embedding.shape
            image_embed = image_features[i]  # B x C x H x W
            image_pe = self.pos_encoding2d(image_embed[:1])
            image_embed = image_embed.flatten(2).permute(
                0, 2, 1
            )  # B x N_image_tokens x C
            image_embed = image_embed + self.interp(
                dense_embeddings,
                image_embed.size(1),
                mode="nearest",  # similar to Feature Pyramid Network (FPN)
            )
            image_pe = image_pe.flatten(2).permute(0, 2, 1)  # B x N_image_tokens x C
            queries, dense_embeddings = cross_attn_layer(
                queries=queries,
                keys=image_embed,
                query_pe=query_pe,
                key_pe=image_pe,
            )

        bs, c, h, w = image_features[-1].shape
        dense_embeddings = dense_embeddings.view(bs, h, w, -1).permute(0, 3, 1, 2)
        # Predict bounding box and mask prompts
        sparse_embeddings = queries

        # Predict bounding box and mask prompts
        interim_mask_output = self.mask_classifier(dense_embeddings)
        dense_embeddings = self.prompt_encoder._embed_masks(
            interim_mask_output
        )  # (B, 256, H//16, W//16)

        # Encode learned box coordinates
        pred_boxes = self.box_regression_head(sparse_embeddings[:, :2])
        learned_box_embeddings = self.prompt_encoder._embed_boxes(
            boxes=pred_boxes * self.prompt_encoder.input_image_size[0]
        )

        sparse_embeddings = torch.cat(
            [learned_box_embeddings, sparse_embeddings[:, 2:]], dim=1
        )

        return (
            sparse_embeddings,
            dense_embeddings,
            interim_mask_output,
            pred_boxes,
        )


@register_prompt_predictor("sam2_high_res_ppn")
class Sam2HighResPPN(HighResPPN):
    pass


@register_prompt_predictor("sam_vitb_high_res_ppn")
class SamViTBHighResPPN(HighResPPN):
    def __init__(
        self,
        *,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        depth: int = 3,
        channel_dims: List[int] = [256, 768, 768],
    ) -> None:
        super().__init__(
            prompt_encoder=prompt_encoder,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            depth=depth,
            channel_dims=channel_dims,
            mask_scale_factor=4,
        )


@register_prompt_predictor("sam_vitl_high_res_ppn")
class SamViTLHighResPPN(HighResPPN):
    def __init__(
        self,
        *,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        depth: int = 3,
        channel_dims: List[int] = [256, 1024, 1024],
    ) -> None:
        super().__init__(
            prompt_encoder=prompt_encoder,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            depth=depth,
            channel_dims=channel_dims,
            mask_scale_factor=4,
        )


@register_prompt_predictor("sam_vith_high_res_ppn")
class SamViTHHighResPPN(HighResPPN):
    def __init__(
        self,
        *,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        depth: int = 3,
        channel_dims: List[int] = [256, 1280, 1280],
    ) -> None:
        super().__init__(
            prompt_encoder=prompt_encoder,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            depth=depth,
            channel_dims=channel_dims,
            mask_scale_factor=4,
        )


@register_prompt_predictor("mobile_sam_high_res_ppn")
class MobileSamHighResPPN(HighResPPN):
    def __init__(
        self,
        *,
        prompt_encoder: nn.Module,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        depth: int = 3,
        channel_dims: List[int] = [256, 320, 320],
    ) -> None:
        super().__init__(
            prompt_encoder=prompt_encoder,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            depth=depth,
            channel_dims=channel_dims,
            mask_scale_factor=4,
        )
