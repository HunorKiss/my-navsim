from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class CameraOnlyConfig:
    """Global CameraOnly config."""

    environment_min_x: float = -32
    environment_max_x: float = 32
    environment_min_y: float = -32
    environment_max_y: float = 32