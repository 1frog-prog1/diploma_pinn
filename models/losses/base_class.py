import torch
import torch.nn as nn
import torch.autograd as autograd
from abc import ABC, abstractmethod

class BasePINNLoss(nn.Module, ABC):
    def __init__(self, equation, u_model):
        super(BasePINNLoss, self).__init__()
        self.equation = equation
        self.u_model = u_model

    def compute_physics_loss(self, residual):
        return torch.mean(residual ** 2)

    def compute_boundary_loss(self, u_bcs, u_pred_bcs):
        return torch.mean((u_pred_bcs - u_bcs) ** 2)

    def compute_initial_loss(self, u_ics, u_pred_ics):
        return torch.mean((u_pred_ics - u_ics) ** 2)

    def compute_data_loss(self, u_data, u_pred_data):
        if u_pred_data is not None and u_data is not None:
            return torch.mean((u_pred_data - u_data) ** 2)
        return torch.tensor(0.0, device=self.get_model_device())
    
    def get_model_device(self):
        return next(self.u_model.parameters()).device

    @abstractmethod
    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        pass