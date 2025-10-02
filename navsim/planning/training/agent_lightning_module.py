from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor
import time, torch

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

         # --- Log model size once ---
        num_params = sum(p.numel() for p in self.agent.parameters()) / 1e6  # in millions
    
    def on_train_start(self):
        """Log model size once training starts (trainer is attached)."""
        self.log("model/num_params_M", self._num_params, prog_bar=True, rank_zero_only=True)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        # --- Measure latency ---
        start = time.time()
        prediction = self.agent.forward(features)
        latency = (time.time() - start) * 1000  # ms

        # --- Compute throughput ---
        batch_size = next(iter(features.values())).size(0)  # take batch size from one feature tensor
        throughput = batch_size / latency if latency > 0 else 0.0

        # --- Compute loss ---
        loss = self.agent.compute_loss(features, targets, prediction)

        # --- Log metrics (averaged per epoch) ---
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/latency_ms", latency, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/throughput_samples_per_sec", throughput, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Log GPU memory (only if CUDA available) ---
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.log(f"{logging_prefix}/gpu_mem_mb", mem_mb, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_train_batch_end(self, outputs, batch, batch_idx):
     lr = self.trainer.optimizers[0].param_groups[0]["lr"]
     self.log("train/lr_step", lr, on_step=True, on_epoch=True, prog_bar=False)
