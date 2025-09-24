import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Dict
from transformers import AutoFeatureExtractor, AutoModel, AutoImageProcessor, SwinModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import StateSE2Index
from navsim.agents.camera_only.cross_attention import CrossAttention
from navsim.agents.camera_only.camera_only_features import BoundingBox2DIndex
from torchvision.transforms.functional import to_pil_image


class CameraOnlyModel(nn.Module):
    def __init__(self, trajectory_sampling: TrajectorySampling, hidden_dim=128, ego_mlp_output_dim=384, num_attention_heads=4):
        """
        Initializes the camera-only model with ViT backbone and Transformer encoder.

        """
        super().__init__()

        # 1. Vision Transformer (ViT) Backbone
        # For Google ViT model, use AutoFeatureExtractor and AutoModel from transformers library
        # Replace with DINO
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dino-vits8")
        self.vit = AutoModel.from_pretrained("facebook/dino-vits8")
        vit_output_dim = self.vit.config.hidden_size  # should be 384 for dino-vits8

        '''
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vit_model) # preprocessing pipeline for the vit model
        self.vit = AutoModel.from_pretrained(vit_model)
        vit_output_dim = 768
        '''

        '''
        # For Swin Transformer model, use AutoImageProcessor and SwinModel from transformers library
        self.feature_extractor = AutoImageProcessor.from_pretrained(vit_model) # preprocessing pipeline for the vit model
        self.vit = SwinModel.from_pretrained(vit_model)
        vit_output_dim = 1024    # Swin produces a 768D feature vector, google produces a 1024D feature vector
        '''
        
        # Feature Fusion
        self._initialize_concatenation_merge_fusion(vit_output_dim, ego_mlp_output_dim, hidden_dim)
        # self._initialize_cross_attention_fusion(vit_output_dim, ego_mlp_output_dim, hidden_dim, num_attention_heads)

        # 2. Ego-Status Feature Extractor
        self.ego_mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ego_mlp_output_dim),
        )

        # 3. Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=vit_output_dim, nhead=8),
            num_layers=2
        )

        # 4. Trajectory Prediction Head (Outputs waypoints)
        self._trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=hidden_dim,
            d_model=vit_output_dim
        )

    def _initialize_concatenation_merge_fusion(self, vit_output_dim, ego_mlp_output_dim, hidden_dim):
        """
        Initializes the feature fusion mechanism using concatenation and MLP.
        """
        fusion_dim = vit_output_dim + ego_mlp_output_dim  # Dimension after concatenation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vit_output_dim),  # Project back to ViT dimension (optional)
        )

    def _initialize_cross_attention_fusion(self, vit_output_dim, ego_mlp_output_dim, hidden_dim, num_attention_heads):
        """
        Initializes the cross-attention mechanism for fusing visual and ego-state features.
        """
        self.cross_attention = CrossAttention(
            query_dim=vit_output_dim,
            key_dim=ego_mlp_output_dim,
            num_heads=num_attention_heads
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(vit_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vit_output_dim)
        )

    def _simple_addition_fusion(self, vit_embedding, status_embedding):
        """
        Feature fusion mechanism using simple addition.
        """
        fused_feature = vit_embedding + status_embedding
        return fused_feature

    def _concatenate_and_fuse_features(self, vit_embedding, status_embedding):
        '''
        Concatenate and fuse visual and ego-state features using MLP.
        '''
        fused_feature = torch.cat((vit_embedding, status_embedding), dim=1) # Concatenate along feature dimension (dim=1)
        fused_feature = self.fusion_mlp(fused_feature)  # (B, 1024)

        return fused_feature

    def _compute_cross_attention_fusion(self, vit_embedding, status_embedding):
        '''
        Compute cross-attention between visual and ego-state features.
        '''
        # Add a sequence dimension of 1 to embeddings for attention
        status_embedding_seq = status_embedding.unsqueeze(1) # (B, 1, hidden_dim)
        vit_embedding_seq = vit_embedding.unsqueeze(1)     # (B, 1, vit_output_dim)

        # Cross-Attention: Visual attends to Ego-State (or vice-versa)
        attended_visual = self.cross_attention(query=vit_embedding_seq, key=status_embedding_seq, value=status_embedding_seq)
        fused_feature = vit_embedding + attended_visual.squeeze(1) # Residual connection but no normalization
        fused_feature = self.fusion_ffn(fused_feature)

        return fused_feature

    def forward(self, features):
        """
        Forward pass through the model.

        :param features: Dictionary containing "camera_feature" and "status_feature".
        :return: Predicted trajectory waypoints.
        """
        camera_input: torch.Tensor = features["camera_feature"]  # (B, C, H, W)
        status_input: torch.Tensor = features["status_feature"]  # (B, 8)

        # 1. Process Camera Features through ViT
        '''
        inputs = self.feature_extractor(camera_input, return_tensors="pt")["pixel_values"].to(camera_input.device)
        vit_outputs = self.vit(inputs)
        vit_embedding = vit_outputs.last_hidden_state[:, 0, :]  # Extract CLS token embedding (B, 1024) 1: a 0 helyett
        '''
        
        resize_transform = T.Resize((224, 224))
        camera_input_resized = resize_transform(camera_input, camera_input)

        pil_images = [to_pil_image(img.cpu()) for img in camera_input_resized]
        vit_embedding = self._extract_vit_embedding(pil_images)

        # 2. Process Ego-Status Features
        status_embedding = self.ego_mlp(status_input)  # (B, 1024)
        
        fused_feature = self._simple_addition_fusion(vit_embedding, status_embedding)
        # fused_feature = self._concatenate_and_fuse_features(vit_embedding, status_embedding)
        # fused_feature = self. _compute_cross_attention_fusion(vit_embedding, status_embedding)

        # If we don’t add the sequence dimension, the tensor shape is (B, 1024) instead of (B, 1, 1024), which will cause an error when passed into the Transformer
        transformed_feature = self.transformer(fused_feature.unsqueeze(1)).squeeze(1)  # (B, vit_output_dim)

        # 5. Trajectory Prediction
        trajectory = self._trajectory_head(transformed_feature)

        return {"trajectory": trajectory}  # (B, num_poses, 3)

        # transzformerek + pytorch doku

    def _extract_vit_embedding(self, image_tensor: torch.Tensor, camera_input) -> torch.Tensor:
        """
        Extract CLS token embedding from ViT.
        :param image_tensor: (B, C, H, W)
        :return: (B, vit_output_dim)
        """
        # inputs = {"pixel_values": image_tensor}
        inputs = self.image_processor(images=image_tensor, return_tensors="pt")
        inputs = {k: v.to(camera_input.device) for k, v in inputs.items()}
        vit_outputs = self.vit(**inputs)
        return vit_outputs.last_hidden_state[:, 0, :]  # CLS token

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: Number of (x,y,θ) poses to predict.
        :param d_ffn: Feed-forward network size.
        :param d_model: Input dimensionality.
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

        
    def forward(self, object_queries) -> torch.Tensor:
        """Predicts trajectory waypoints."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * 3.14  # Normalize heading
        return poses  # (B, num_poses, 3)
