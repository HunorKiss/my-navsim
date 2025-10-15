import math
import numpy as np
from typing import Dict
import cv2
from enum import IntEnum
from typing import Any, Dict, List, Tuple
import numpy.typing as npt

import torch
from torchvision import transforms
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.enums import BoundingBoxIndex
from navsim.agents.camera_only.camera_only_config import CameraOnlyConfig

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
        features["front_camera_feature"] = self._get_front_camera_feature(agent_input)
        # features["back_camera_feature"] = self._get_back_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_pose, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_front_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
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
    
    def _get_back_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        # print(agent_input.cameras)

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l2 = cameras.cam_l2.image[28:-28, 416:-416]
        b0 = cameras.cam_b0.image[28:-28]
        r2 = cameras.cam_r2.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([r2, b0, l2], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))

        # OpenCV gives BGR, convert to RGB
        # l-----> ez változtatás a TransFuserhez képest
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

class CameraOnlyTargetBuilder(AbstractTargetBuilder):
    """Output target builder for CameraOnly."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: CameraOnlyConfig
    ):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: camera-only configuration.
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "camera_only_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses).poses
        )

        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        
        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels
        }
    
    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_area(x: float, y: float, config: CameraOnlyConfig) -> bool:
            """
            Check if a point (x, y) is inside the square area and within the front camera FOV.
            """
            # 1. Square area check
            inside_square = (
                config.environment_min_x <= x <= config.environment_max_x
                and config.environment_min_y <= y <= config.environment_max_y
            )

            if not inside_square:
                return False

            # 2. Front camera FOV check (140 degrees)
            fov_rad = math.radians(140)  # 140 degrees in radians
            angle = math.atan2(y, x)     # angle from ego to agent (0 = straight ahead)
            if abs(angle) > fov_rad / 2:
                return False

            return True

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_area(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array(
                        [box_x, box_y, box_heading, box_length, box_width],
                        dtype=np.float32,
                    )
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)
    

class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)