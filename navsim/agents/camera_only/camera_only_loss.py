from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

def camera_only_loss( targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]):
    """
    Helper function calculating complete loss of CameraOnly
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :return: combined loss value
    """

    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    loss = trajectory_loss
    
    return loss