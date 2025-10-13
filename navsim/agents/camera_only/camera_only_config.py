from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class CameraOnlyConfig:
    """Global CameraOnly config."""

    aux_tasks_enabled: bool = False

    # detection
    environment_min_x: float = -32
    environment_max_x: float = 32
    environment_min_y: float = -32
    environment_max_y: float = 32

    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    bev_semantic_weight: float = 1.0
    agent_box_weight: float = 1.0

    # Transformer
    tf_d_model: int = 1024
    tf_d_ffn: int = 1024

    tf_num_head: int = 8
    tf_dropout: float = 0.0
    tf_num_layers: int = 3

    # detection
    num_bounding_boxes: int = 30

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    # BEV mapping
    num_bev_classes = 7
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: (
            "linestring",
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        ),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_resolution_width = 256
    bev_resolution_height = 256

    bev_pixel_width: int = bev_resolution_width
    bev_pixel_height: int = bev_resolution_height // 2
    bev_pixel_size: float = 0.25

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)
    
    @property
    def bev_radius(self) -> float:
        values = [
            self.environment_min_x,
            self.environment_max_x,
            self.environment_min_y,
            self.environment_max_x,
        ]
        return max([abs(value) for value in values])