from torch import nn
from base_class import BasePINN
from losses import DB_PINN_Loss

class DB_PINN(BasePINN):
    def __init__(
        self,
        input_dim, output_dim, hidden_layers,
        equation,
        activation=nn.Tanh(), scaling_function=None,
        rff_features=0, rff_sigma=1.0, seed=None
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            equation=equation,
            loss_class=DB_PINN_Loss,
            activation=activation,
            scaling_function=scaling_function,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            seed=seed
        )