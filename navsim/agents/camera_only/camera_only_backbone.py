import copy
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

# Assuming TransfuserConfig, GPT, SelfAttention, Block, etc., are available from your provided code snippet.
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
# Re-importing necessary custom components for clarity (assuming they are in the same scope)
from navsim.agents.transfuser.transfuser_backbone import GPT, SelfAttention, Block, MultiheadAttentionWithAttention, TransformerDecoderLayerWithAttention, TransformerDecoderWithAttention


class CameraOnlyBackbone(nn.Module):
    """
    Implements a single-modality (Image-only) backbone for TransFuser-style
    architecture with self-attention modules at multiple scales.
    """

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config

        # --- 1. Image Encoder (Pretrained) ---
        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)

        # --- LiDAR components are removed ---
        # Instead of lidar_channel_to_img/img_channel_to_lidar, the 'transformers' 
        # will now perform self-attention on the image features only.
        
        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1
        
        # Determine the feature channel dimensions for the 4 CNN blocks
        num_chs = [
            self.image_encoder.feature_info.info[start_index + i]["num_chs"]
            for i in range(4)
        ]

        # --- 2. Image-Only Transformer Modules (Self-Attention) ---
        # We reuse the GPT structure but it will only process one input (image)
        # This implementation requires a slight modification to the original GPT or a new module.
        # For simplicity and to reuse the code, we'll create a single-input self-attention module.
        self.self_attention_blocks = nn.ModuleList(
            [
                # We need a SelfAttentionModule that handles only a single input (image)
                # and returns a single output. We will simplify the GPT/Block usage here.
                # A full block ensures we have the LN-Attn-LN-MLP structure.
                nn.Sequential(
                    nn.LayerNorm(n_embd),
                    SelfAttention(
                        n_embd,
                        config.n_head,
                        config.attn_pdrop,
                        config.resid_pdrop,
                    ),
                    nn.Dropout(config.embd_pdrop),
                    nn.LayerNorm(n_embd),
                    nn.Sequential(
                        nn.Linear(n_embd, config.block_exp * n_embd),
                        nn.ReLU(True),
                        nn.Linear(config.block_exp * n_embd, n_embd),
                        nn.Dropout(config.resid_pdrop),
                    )
                )
                for n_embd in num_chs
            ]
        )
        
        self.ln_f = nn.ModuleList([nn.LayerNorm(n_embd) for n_embd in num_chs])

        # --- Final Feature Output Configuration ---
        self.num_features = num_chs[-1] # Final output size from the encoder

        # Ideiglenes ellenőrzés a CameraOnlyBackbone.__init__ végén:
        assert num_chs == [64, 128, 256, 512], f"Expected channels [64, 128, 256, 512], but got {num_chs}"

    def forward(self, image: torch.Tensor, lidar=None):
        """
        Image feature extraction and multi-scale self-attention.
        Args:
            image (torch.Tensor): The RGB image input.
            lidar (None): Placeholder, unused in this model.
        """
        image_features = image

        # Generate an iterator for all the layers in the network
        image_layers = iter(self.image_encoder.items())

        # Stem layer
        if len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

        # Loop through the 4 CNN blocks + Self-Attention
        for i in range(4):
            # 1. CNN Block forward
            image_features = self._forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            
            # 2. Self-Attention Fusion (Residue connection is implicit in the SelfAttentionModule)
            image_features = self._self_attend_features(image_features, i)

        # Final pooling and flattening
        image_feature_grid = image_features # The image features grid for semantic/depth prediction

        # Final global feature vector
        global_feature = self.global_pool_img(image_features)
        global_feature = torch.flatten(global_feature, 1) # This is the 512-dim vector in the Transfuser diagram

        # We return None for BEV features since this is an image-only model
        return None, global_feature, image_feature_grid

    def _forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network.
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _self_attend_features(self, image_features, layer_idx):
        """
        Perform a Self-Attention block using Image features.
        :param image_features: Features from the image branch
        :param layer_idx: Transformer layer index.
        :return: Image features with added self-attention context.
        """
        # 1. Downsample features to anchor grid (e.g., 22x5)
        image_embd_layer = self.avgpool_img(image_features)
        
        # 2. Prepare for Self-Attention (B, C, H, W) -> (B, N, C)
        bz, c, h, w = image_embd_layer.shape
        image_tensor = image_embd_layer.permute(0, 2, 3, 1).contiguous().view(bz, -1, c) # (B, H*W, C)

        # 3. Add Positional Embedding/Encoding if desired (omitted for simplicity, but can be added in a full implementation)
        # 4. Self-Attention Block
        # We simulate the inner workings of Block(LN-Attn-LN-MLP) to get the residual connection
        token_embeddings = image_tensor
        x = self.ln_f[layer_idx](token_embeddings) # Pre-LN for Transformer Block
        
        # --- Self-Attention ---
        # The structure is similar to the self-attention part of the original Block
        x_attn = self.self_attention_blocks[layer_idx][0](token_embeddings) # LN1
        x_attn = self.self_attention_blocks[layer_idx][1](x_attn) # Attn
        x_attn = token_embeddings + x_attn # Residual connection 1

        # --- MLP ---
        x_mlp = self.self_attention_blocks[layer_idx][3](x_attn) # LN2
        x_mlp = self.self_attention_blocks[layer_idx][4](x_mlp) # MLP
        x_out = x_attn + x_mlp # Residual connection 2

        # 5. Reshape and upsample (B, N, C) -> (B, C, H, W)
        image_features_out = (
            x_out.view(bz, h, w, c).permute(0, 3, 1, 2).contiguous()
        )
        
        image_features_out = F.interpolate(
            image_features_out,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # 6. Add to original features (Residual connection from the paper's diagram)
        image_features = image_features + image_features_out

        return image_features