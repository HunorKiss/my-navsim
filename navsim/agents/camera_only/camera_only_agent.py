import numpy as np
from typing import Any, List, Dict, Optional, Union
import cv2

import torch
from navsim.agents.camera_only.camera_only_features import CameraOnlyFeatureBuilder, CameraOnlyTargetBuilder
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.camera_only.camera_only_model import CameraOnlyModel
from navsim.agents.camera_only.camera_only_loss import camera_only_loss
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from navsim.agents.camera_only.camera_only_config import CameraOnlyConfig

class CameraOnlyAgent(AbstractAgent):
    """Agent implementing a camera-only model."""

    requires_scene = False

    def __init__(
        self,
        config: CameraOnlyConfig,
        lr: float,
        lr_decay_step: float,  # Example: decay every epoch
        lr_decay_gamma: float,  # Example: decay by a factor of 0.95
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes the agent interface for CameraOnly.
        :param trajectory_sampling: trajectory sampling specification.
        :param lr: learning rate during training.
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr
        self._lr_decay_step = lr_decay_step
        self._lr_decay_gamma = lr_decay_gamma
        self._checkpoint_path = checkpoint_path
        self._trajectory_sampling = trajectory_sampling

        self._camera_only_model = CameraOnlyModel(self._trajectory_sampling, self._config)

        for param in self._camera_only_model.vit.parameters():
            param.requires_grad = True  # True = fine-tune, False = freeze

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._camera_only_model.to_empty(device=device)

        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=device)[
            "state_dict"
        ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [CameraOnlyTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [CameraOnlyFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        model_device = next(self._camera_only_model.parameters()).device
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to(model_device)
                
        return self._camera_only_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return camera_only_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        """
        optimizer = torch.optim.Adam(
            list(self._camera_only_model.ego_mlp.parameters()) +
            list(self._camera_only_model.vit.parameters()) +
            list(self._camera_only_model.fusion_mlp.parameters()) +
            list(self._camera_only_model.transformer.parameters()) +
            list(self._camera_only_model._trajectory_head.parameters()) +
            list(self._camera_only_model._agent_head.parameters()),
            lr=self._lr
        )
        scheduler = StepLR(optimizer, step_size=self._lr_decay_step, gamma=self._lr_decay_gamma)
        """

        optimizer = torch.optim.Adam(self._camera_only_model.parameters(), lr=self._lr)
        # Cosine annealing scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,      # első félperiódus = 10 epoch
            T_mult=2,    # a következő periódus hossza kétszerese az előzőnek
            eta_min=1e-5)  # a minimális LR

        return {"optimizer": optimizer, "lr_scheduler": scheduler}