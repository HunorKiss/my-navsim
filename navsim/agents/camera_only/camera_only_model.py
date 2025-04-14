import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoFeatureExtractor, AutoModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import StateSE2Index

class CameraOnlyModel(nn.Module):
    def __init__(self, trajectory_sampling: TrajectorySampling, vit_model="google/vit-base-patch16-224-in21k", hidden_dim=256):
        """
        Initializes the camera-only model with ViT backbone, Transformer encoder, and GRU decoder.
        
        :param vit_model: Name of the pretrained ViT model.
        :param hidden_dim: Hidden size for Transformer and GRU layers.
        :param num_waypoints: Number of waypoints to predict.
        """
        super().__init__()

        # 1. Vision Transformer (ViT) Backbone
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vit_model) # preprocessing pipeline for the vit model
        self.vit = AutoModel.from_pretrained(vit_model)
        vit_output_dim = 768  # DINO-ViT produces a 768D feature vector

        # 2. Ego-Status Feature Extractor
        self.ego_mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vit_output_dim),  # Project to match ViT feature size
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

    def forward(self, features):
        """
        Forward pass through the model.

        :param features: Dictionary containing "camera_feature" and "status_feature".
        :return: Predicted trajectory waypoints.
        """
        camera_input: torch.Tensor = features["camera_feature"]  # (B, C, H, W)
        status_input: torch.Tensor = features["status_feature"]  # (B, 8)

        # 1. Process Camera Features through ViT
        inputs = self.feature_extractor(camera_input, return_tensors="pt")["pixel_values"].to(camera_input.device)
        vit_outputs = self.vit(inputs)
        vit_embedding = vit_outputs.last_hidden_state[:, 0, :]  # Extract CLS token embedding (B, 768) 1: a 0 helyett

        # 2. Process Ego-Status Features
        status_embedding = self.ego_mlp(status_input)  # (B, 768)

        # 3. Feature Fusion
        fused_feature = vit_embedding + status_embedding  # (B, 768) concat a seq dimenzió mentén

        # 4. Temporal Modeling with Transformer
        fused_feature = fused_feature.unsqueeze(1)  # Add sequence dimension (B, 1, 768) valszeg nem kell
        # If we don’t add the sequence dimension, the tensor shape is (B, 768) instead of (B, 1, 768), which will cause an error when passed into the Transformer
        transformed_feature = self.transformer(fused_feature)  # (B, 1, 768)

        # 5. Trajectory Prediction
        trajectory = self._trajectory_head(transformed_feature)

        return {"trajectory": trajectory}  # (B, num_poses, 3)

        # transzformerek + pytorch doku

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
