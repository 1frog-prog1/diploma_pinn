from torch import nn
from base_class import BasePINN
from losses import PINN_Loss

class PirateNetPINN(BasePINN):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        equation,
        loss_class=PINN_Loss,
        activation=nn.Tanh(),
        scaling_function=None,
        rff_features=0,
        rff_sigma=1.0,
        seed=None,
        num_blocks=3,
        loss_kwargs=None
    ):
        model_kwargs = {"num_blocks" : num_blocks}
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            equation=equation,
            loss_class=loss_class,
            activation=activation,
            scaling_function=scaling_function,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            seed=seed,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs
        )