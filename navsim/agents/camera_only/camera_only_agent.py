import numpy as np
from typing import Any, List, Dict, Optional, Union
import cv2

import torch
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.camera_only.camera_only_model import CameraOnlyModel
from navsim.agents.camera_only.camera_only_loss import camera_only_loss
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

class CameraOnlyAgent(AbstractAgent):
    """Agent implementing a camera-only model."""

    requires_scene = False

    def __init__(
        self,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        lr_decay_step: int = 3,  # Example: decay every 3 epochs
        lr_decay_gamma: float = 0.25,  # Example: decay by a factor of 0.1
    ):
        """
        Initializes the agent interface for CameraOnly.
        :param trajectory_sampling: trajectory sampling specification.
        :param lr: learning rate during training.
        """
        super().__init__(trajectory_sampling)

        self._lr = lr
        self._lr_decay_step = lr_decay_step
        self._lr_decay_gamma = lr_decay_gamma
        self._checkpoint_path = checkpoint_path
        self._trajectory_sampling = trajectory_sampling

        self._camera_only_model = CameraOnlyModel(self._trajectory_sampling)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
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
        return [CameraOnlyTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [CameraOnlyFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._camera_only_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return camera_only_loss(predictions, targets)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        optimizer = torch.optim.Adam(
            list(self._camera_only_model.ego_mlp.parameters()) +
            list(self._camera_only_model.vit.parameters()) +
            list(self._camera_only_model.fusion_mlp.parameters()) +
            list(self._camera_only_model.transformer.parameters()) +
            list(self._camera_only_model._trajectory_head.parameters()),
            lr=self._lr
        )
        scheduler = StepLR(optimizer, step_size=self._lr_decay_step, gamma=self._lr_decay_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class CameraOnlyFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of CameraOnly."""

    def __init__(self):
        """Initializes the feature builder."""
        return

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "camera_only_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}
        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_pose, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        print(agent_input.cameras)

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image


class CameraOnlyTargetBuilder(AbstractTargetBuilder):
    """Output target builder for CameraOnly."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "camera_only_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses).poses
        return {"trajectory": torch.tensor(future_trajectory)}