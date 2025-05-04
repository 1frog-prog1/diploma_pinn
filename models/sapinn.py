import torch
import torch.nn as nn
from .utils.mlp import MLP
from .utils.masks import PolynomialMask, SigmoidMask
from .utils.losses import SA_PINN_Loss, PINN_Loss

# Реализация SA-PINN
class SA_PINN(nn.Module):
    def __init__(
            self, input_dim, output_dim, 
            hidden_layers, equation,
            points_pde=None,
            points_ics=None, points_bcs=None,
            mask_class=PolynomialMask,
            activation=nn.Tanh(), scaling_function=None,
            rff_features=0, rff_sigma=1.0, seed=None
        ):
        super(SA_PINN, self).__init__()
        self.u_model = MLP(
            input_dim, hidden_layers, output_dim, 
            activation, scaling_function, 
            rff_features, rff_sigma, seed 
        )
        self.equation = equation
        
        self.loss_class = SA_PINN_Loss(
            points_pde=points_pde,
            points_ics=points_ics,
            points_bcs=points_bcs,
            equation=equation,
            u_model=self.u_model,
            mask_class=mask_class
        )

    def forward(self, x):
        return self.u_model(x)
    
    @property
    def lambdas(self):
        return [self.loss_class.lambdas_pde,
                self.loss_class.lambdas_ics,
                self.loss_class.lambdas_bcs]
    
    @property
    def lambdas_pde(self):
        return self.loss_class.lambdas_pde

    @property
    def lambdas_ics(self):
        return self.loss_class.lambdas_ics

    @property
    def lambdas_bcs(self):
        return self.loss_class.lambdas_bcs

    # Loss function
    def loss(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        """
        Computes the loss function for SA-PINN.

        Parameters:
        - x_data: input data for training (tensor).
        - u_data: true values for the data (tensor).
        - x_pde: input data for the physical equation (tensor).
        - x_ics: input data for initial conditions (tensor).
        - x_bcs: input data for boundary conditions (tensor).

        Returns:
        - Total loss (sum of data loss and physics loss).
        """
        return self.loss_class(x_pde, x_ics, x_bcs, x_data, u_data)
