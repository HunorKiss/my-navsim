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
from navsim.agents.camera_only.camera_only_config import CameraOnlyConfig
from navsim.agents.camera_only.camera_only_backbone import CameraOnlyBackbone
from torchvision.transforms.functional import to_pil_image


class CameraOnlyModel(nn.Module):
    """
    Torch module for a Camera-Only model that uses the Transfuser architecture 
    with image self-attention and final fusion with ego status via a Transformer.
    """

    def __init__(self, trajectory_sampling: TrajectorySampling, config: CameraOnlyConfig):
        """
        Initializes the CameraOnlyModel torch module.
        """

        super().__init__()

        self._query_splits = [
            1, # Trajectory query
            config.num_bounding_boxes, # Agent queries
        ]

        self._config = config
        # Use the new CameraOnlyBackbone
        self._backbone = CameraOnlyBackbone(config)
        
        # The final global image feature (e.g., 1512-dim) needs to be projected to d_model.
        self._image_downscale = nn.Linear(self._backbone.num_features, config.tf_d_model)

        # The key/value embedding is only for the status/auxiliary tokens
        self._keyval_embedding = nn.Embedding(1 + 1, config.tf_d_model) # Ego Status + Trajectory Query
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # Status encoding remains the same (projects status to d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2 + 3, config.tf_d_model)

        # Removed BEV semantic head as we don't have the BEV features, 
        # or it must be replaced with a BEV prediction module from the Image grid.
        # Assuming for this implementation, we drop the BEV map outputs.
        # We will keep the trajectory and agent heads.

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Final fusion transformer remains the same
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        
        # NOTE: If you need to keep the BEV semantic head, you would need 
        # a module to project the image_feature_grid into the BEV space (e.g., using a separate Transformer/MLP).
        self._bev_semantic_head = None 
        if self._config.use_bev_semantic:
            print("Warning: BEV semantic head requires dedicated image-to-BEV projection, which is omitted here.")


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["front_camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        # Backbone output: (bev_feature_upscale, global_feature, image_feature_grid)
        # The first output (bev_feature_upscale) will be None, the second (global_feature) 
        # is the final 1512-dim vector.
        _, global_image_feature, _ = self._backbone(camera_feature)

        # 1. Image features to Transformer dimension (d_model)
        image_keyval = self._image_downscale(global_image_feature)[:, None] # (B, 1, d_model)

        # 2. Ego status encoding
        status_encoding = self._status_encoding(status_feature)[:, None] # (B, 1, d_model)
        
        # 3. Final Key/Value (KV) for the Transformer Decoder
        # The keyval memory now consists of the global Image Feature + Ego Status
        keyval = torch.concatenate([image_keyval, status_encoding], dim=1) # (B, 2, d_model)
        
        # Add a learned positional embedding to distinguish Image from Status
        keyval += self._keyval_embedding.weight[None, :keyval.shape[1], :]

        # 4. Query and Decoder Forward Pass
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1) # Trajectory + Agents
        query_out = self._tf_decoder(query, keyval)

        # 5. Prediction Heads
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        trajectory = self._trajectory_head(trajectory_query)
        output: Dict[str, torch.Tensor] = {"trajectory": trajectory}

        if self._config.aux_tasks_enabled:
            agents = self._agent_head(agents_query)
            output.update(agents)

        return output

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
