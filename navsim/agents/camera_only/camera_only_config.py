from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class CameraOnlyConfig:
    """Global CameraOnly config."""

    aux_tasks_enabled: bool = True
    use_bev_semantic: bool = False

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

    # ===============================================
    # TRANSZFORMÁTOR PARANÉTEREK (Mindkét típushoz)
    # ===============================================
    
    # A VÉGSŐ FÚZIÓS DEKÓDER paraméterei (TransfuserModel)
    tf_d_model: int = 384
    tf_d_ffn: int = 1024
    tf_num_head: int = 4
    tf_num_layers: int = 4
    tf_dropout: float = 0.1

    # A BACKBONE BELSŐ MODULJAINAK (GPT/Block) paraméterei
    # Az n_head azonos a tf_num_heads-szel, de a GPT a régi nevét várhatja
    n_head: int = 4                            # Attention fejek száma (a belső backbone-hoz)
    n_layer: int = 4                           # A GPT modulban lévő Block rétegek száma
    block_exp: int = 4                         # Az MLP kiterjesztési faktora a Block-ban

    # Dropout paraméterek
    attn_pdrop: float = 0.1                    # Attention layer dropout
    resid_pdrop: float = 0.1                   # Residual layer dropout (MLP-ben és Attention-ben)
    embd_pdrop: float = 0.1                    # Embedding dropout

    # GPT súly inicializálási paraméterei (a Block osztályban használatos)
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0
    
    # ===============================================
    
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    # ENCODER PARAMÉTEREK
    image_architecture: str = "resnet34"
    img_vert_anchors: int = 8                  # 256 / 32 = 8. A ResNet kimeneti felbontásához
    img_horz_anchors: int = 32                 # 1024 / 32 = 32. A ResNet kimeneti felbontásához