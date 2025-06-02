import torch
import torch.nn as nn
import torch.autograd as autograd
from .mlp import MLP
from abc import ABC, abstractmethod

class BasePINN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        equation,
        loss_class,
        activation=nn.Tanh(),
        scaling_function=None,
        rff_features=0,
        rff_sigma=1.0,
        seed=None,
        loss_kwargs=None  # аргументы, специфичные для лосса
    ):
        super(BasePINN, self).__init__()
        self.equation = equation
        self.u_model = MLP(
            input_dim, hidden_layers, output_dim,
            activation, scaling_function,
            rff_features, rff_sigma, seed
        )

        loss_kwargs = loss_kwargs or {}
        self.loss_fn = loss_class(equation=equation, u_model=self.u_model, **loss_kwargs)

    def forward(self, x):
        return self.u_model(x)

    def loss(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        return self.loss_fn(x_pde, x_ics, x_bcs, x_data, u_data)

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