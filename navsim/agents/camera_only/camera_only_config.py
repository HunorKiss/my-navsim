from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class CameraOnlyConfig:
    """Global CameraOnly config."""

    # detection
    environment_min_x: float = -32
    environment_max_x: float = 32
    environment_min_y: float = -32
    environment_max_y: float = 32

    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0

    # Transformer
    tf_d_model: int = 384
    tf_d_ffn: int = 1024

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9