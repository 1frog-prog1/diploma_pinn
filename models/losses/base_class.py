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
        return torch.tensor(0.0, device=self.__get_model_device())
    
    def __get_model_device(self):
        return next(self.u_model.parameters()).device
    
    def get_distinct_losses(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        t_pde, x_pde = x_pde[:, 0:1], x_pde[:, 1:]
        phys_res = self.equation.residual_function(self.u_model, t_pde, x_pde)
        phys_loss = self.compute_physics_loss(phys_res)

        t_bcs = x_bcs[:, 0:1]
        u_bcs = self.equation.boundary_conditions(t_bcs)
        u_pred_bcs = self.u_model(x_bcs)
        bcs_loss = self.compute_boundary_loss(u_bcs, u_pred_bcs)

        x_ics_split = x_ics[:, 1:]
        u_ics = self.equation.initial_conditions(x_ics_split)
        u_pred_ics = self.u_model(x_ics)
        ics_loss = self.compute_initial_loss(u_ics, u_pred_ics)

        data_loss = self.compute_data_loss(u_data, self.u_model(x_data) if x_data is not None and u_data is not None else None)

        return {
            "physics_loss": phys_loss,
            "boundary_loss": bcs_loss,
            "initial_loss": ics_loss,
            "data_loss": data_loss
        }

    @abstractmethod
    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        pass