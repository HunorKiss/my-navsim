import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List, Union
from transformers import AutoModel, AutoImageProcessor
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import StateSE2Index
import torchvision.transforms as T
from typing import Dict, Tuple
from navsim.common.enums import StateSE2Index
from navsim.agents.camera_only.cross_attention import CrossAttention
from navsim.agents.camera_only.camera_only_features import BoundingBox2DIndex
from navsim.agents.camera_only.camera_only_config import CameraOnlyConfig
from torchvision.transforms.functional import to_pil_image
from transformers import pipeline
from transformers.image_utils import load_image

# --- KONSTANS BEÁLLÍTÁSOK ---
DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
D_MODEL_TARGET = 1024  # A stabil Transzformer dimenzió


# --- SEGÉD OSZTÁLYOK (Fejek) ---
class BEVSemanticHead(nn.Module):
    """Auxiliary BEV Semantic Head: Többlépcsős (Multi-Step) Upsampling"""
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        
        self.bev_target_height = 128
        self.bev_target_width = 256
        
        # A bemenetünk most 14x14 (DINO patch-ek)
        
        self.decoder = nn.Sequential(
            # Lépés 1: Dimenziócsökkentés
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1), # Pl. 1024 -> 512 csatorna
            nn.ReLU(),

            # Lépés 2: Felskálázás (14 -> 28)
            nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Lépés 3: Felskálázás (28 -> 56)
            nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Lépés 4: FPN-szerű utolsó lépés (56 -> 112)
            nn.ConvTranspose2d(d_model // 8, d_model // 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Végleges felskálázás és kimeneti osztályok (112 -> 128 / 256-ra kényszerítve)
            nn.Conv2d(d_model // 16, num_classes, kernel_size=1)
        )
        
        # NOTE: A kimenet méretét kényszerítjük a legközelebbi 128x256-ra
        self.final_upsample = nn.Upsample(
            size=(self.bev_target_height, self.bev_target_width),
            mode='bilinear', 
            align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.decoder(x)
        return self.final_upsample(x)


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


# --- FŐ MODELL ---
class CameraOnlyModel(nn.Module):
    def __init__(self, trajectory_sampling: TrajectorySampling, config: CameraOnlyConfig):
        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes
        ]  # 1 trajectory + N agents
        
        self._config = config
        self._aux_tasks_enabled = config.aux_tasks_enabled
        
        # 1. DINO Vizuális Encoder (4096 dimenzió)
        self._processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        self._model_vit = AutoModel.from_pretrained(DINO_MODEL_NAME)
        vit_output_dim = self._model_vit.config.hidden_size  # 4096

        # 2. Projekció: 4096 -> 1024 (Dimenziócsökkentés)
        self._feature_projector = nn.Linear(vit_output_dim, D_MODEL_TARGET) 
        
        # 3. Ego Status Encoder (MLP)
        self._status_encoding = nn.Linear(11, D_MODEL_TARGET)  # 8 -> 1024

        # 4. Transzformer Dekóder (BEV Fúzió és Lekérdezés)
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL_TARGET, 
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._transformer_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        # 5. Queries (Trajektória és Agent)
        self._keyval_embedding = nn.Embedding(14**2 + 1, D_MODEL_TARGET)
        self._query_embedding = nn.Embedding(sum(self._query_splits), D_MODEL_TARGET)

        # 6. Fejek Inicializálása
        self._trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn, 
            d_model=D_MODEL_TARGET
        )
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=D_MODEL_TARGET
        )
        self._bev_semantic_head = BEVSemanticHead(
            d_model=D_MODEL_TARGET, num_classes=config.num_bev_classes
        )

    # --- Segédfüggvény: DINO feature kinyerése ---
    def _extract_dino_features(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        device = self._model_vit.device
        # print(device)

        # Négyzetes bemenetre kényszerítés (224x224)
        resize_transform = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        image_tensor = resize_transform(image_tensor)

        inputs = self._processor(images=image_tensor, return_tensors="pt")
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Futtatás (GRADIENSEKKEL!)
        outputs = self._model_vit(**inputs)

        cls_token = outputs.last_hidden_state[:, 0, :]       
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]  # 4 register token + 196 patch token 
        return cls_token, patch_embeddings

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        camera_feature: torch.Tensor = features["front_camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        # 1. DINO Feature Kinyerés és Projekció
        _, patch_embeddings_4096 = self._extract_dino_features(camera_feature)
        patch_embeddings_1024 = self._feature_projector(patch_embeddings_4096)

        # 2. Ego Kódolás
        status_encoding_1024 = self._status_encoding(status_feature)
        global_status_token = status_encoding_1024.unsqueeze(1) 

        # 3. Memória Képzés
        keyval = torch.cat([patch_embeddings_1024, global_status_token], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]
        
        # 4. Dekóder Futtatása
        batch_size = camera_feature.shape[0]
        queries = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._transformer_decoder(queries, keyval) 
        
        # 5. Fejek (Kimeneti szétválasztás)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        
        # Trajektória (Trajectory Head a Trajektória Query-re)
        trajectory = self._trajectory_head(trajectory_query.squeeze(1))
        output: Dict[str, torch.Tensor] = {"trajectory": trajectory}
        
        # Agent és Bev Szemantika (ha engedélyezve)
        if self._aux_tasks_enabled:
            # Agent (Detektálás)
            agents = self._agent_head(agents_query)
            output.update(agents)

            # BEV Szemantika (Auxiliary Task)
            ''' N_PATCHES_SPATIAL = 196
            bev_feature_map = keyval[:, :-1, :]    
            bev_feature_map_clean = bev_feature_map[:, :N_PATCHES_SPATIAL, :]
            bev_size = 14        
            bev_feature_map_2d = bev_feature_map_clean.permute(0, 2, 1).reshape(batch_size, D_MODEL_TARGET, bev_size, bev_size)
            output["bev_semantic_map"] = self._bev_semantic_head(bev_feature_map_2d) '''

        return output