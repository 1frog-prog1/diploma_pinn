import torch
import torch.nn as nn
from .utils.mlp import MLP
from .utils.losses import PINN_Loss

class PINN(nn.Module):
    def __init__(
            self, 
            input_dim, output_dim, hidden_layers, 
            equation, activation=nn.Tanh(), scaling_function=None,
            rff_features=0, rff_sigma=1.0, seed=None
        ):
        super(PINN, self).__init__()
        self.equation = equation
        self.u_model = MLP(input_dim, hidden_layers, output_dim, 
                           activation, scaling_function, 
                           rff_features, rff_sigma, seed)
        self.loss_fn = PINN_Loss(equation, self.u_model)

    def forward(self, x):
        return self.u_model(x)

    def loss(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        return self.loss_fn(x_pde, x_ics, x_bcs, x_data, u_data)
