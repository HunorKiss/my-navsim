import numpy as np
from typing import Any, List, Dict, Optional, Union
import torch
from navsim.agents.camera_only.camera_only_features import CameraOnlyFeatureBuilder, CameraOnlyTargetBuilder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR, CosineAnnealingWarmRestarts
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.camera_only.camera_only_model import CameraOnlyModel
from navsim.agents.camera_only.camera_only_loss import camera_only_loss
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.camera_only.camera_only_config import CameraOnlyConfig


class CameraOnlyAgent(AbstractAgent):
    """Agent implementing a robust DINO-BEV-Query model."""

    def __init__(
        self,
        config: CameraOnlyConfig,
        lr: float,
        lr_decay_step: float,
        lr_decay_gamma: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr
        self._lr_decay_step = lr_decay_step
        self._lr_decay_gamma = lr_decay_gamma
        self._checkpoint_path = checkpoint_path
        self._trajectory_sampling = trajectory_sampling
        self.weight_decay = 1e-4

        self._camera_only_model = CameraOnlyModel(self._trajectory_sampling, self._config)

        for param in self._camera_only_model._model_vit.parameters():
            param.requires_grad = False # <-- Ez fagyasztja a ViT paramétereit!

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Súlybetöltés, optimalizálva a DDP/GPU környezethez."""

        # 1. Betöltjük a súlyokat a CPU-ra (DDP standard)
        #state_dict: Dict[str, Any] = torch.load(
        #    self._checkpoint_path, 
        #    map_location=torch.device("cpu") 
        #)["state_dict"]

        # 2. Szűrjük a kulcsokat a Transfuser logikája szerint
        #model_state_dict = self._camera_only_model.state_dict()
        #new_state_dict = {}
        
        #for k, v in state_dict.items():
        #    model_key = k.replace("agent.", "")
        #    
        #    # Súlyok CPU-n átadása a DDP-nek
        #    if model_key in model_state_dict:
        #        new_state_dict[model_key] = v
        
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
                
        # 3. Betöltjük az állapotot
        self.load_state_dict(state_dict, strict=False)


    def get_sensor_config(self) -> SensorConfig:
        """Kamera-Only beállítások a DINO-hoz (Transfuser alapján)."""
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False, 
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """CameraOnly Target Builder (BEV Semantikával) használata."""
        return [CameraOnlyTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """CameraOnly Feature Builder használata."""
        return [CameraOnlyFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Modell futtatása és bemenet áthelyezése a modell eszközére."""

        # Nem helyes, ha a bemeneteket itt helyezzük át, mert a DDP már kezeli az eszközöket.
        # model_device = next(self._camera_only_model.parameters()).device
        # for key, value in features.items():
        #    if isinstance(value, torch.Tensor):
        #        features[key] = value.to(model_device)
        
        return self._camera_only_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Kameramodell veszteségfüggvényének hívása (BEV Semantikával)."""
        # Feltételezve, hogy a camera_only_loss most már tartalmazza a BEV Semantic Loss-t
        return camera_only_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Optimalizáló beállítása és Cosine Annealing scheduler."""

        # Kinyeri csak azokat a paramétereket, amelyeknél a requires_grad TRUE (transzformer, fejek és MLP-k)
        learnable_params = filter(
            lambda p: p.requires_grad, self._camera_only_model.parameters()
        )

        # Az optimizer csak a transzformert, fejeket, és MLP-t fogja frissíteni.
        optimizer = torch.optim.Adam(learnable_params, lr=self._lr, weight_decay=self._weight_decay)
        
        # Cosine Annealing (a nagy modellek stabilizálásához szükséges)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Első periódus 5 epoch
            T_mult=3,   # Periódus hosszának exponenciális növelése
            eta_min=1e-6) # Alacsony minimális Learning Rate

        return {"optimizer": optimizer, "lr_scheduler": scheduler}