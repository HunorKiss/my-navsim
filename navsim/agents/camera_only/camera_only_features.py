import numpy as np
from typing import Dict
import cv2

import torch
from torchvision import transforms
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


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

        # print(agent_input.cameras)

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))

        # OpenCV gives BGR, convert to RGB
        # l-----> ez változtatás a TransFuserhez képest
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

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