from torch import nn
from .base_class import BasePINN
from models.losses import PINN_Loss
from models.base_models import PirateNet

class PirateNetPINN(BasePINN):
    def __init__(
        self,
        input_dim,
        output_dim,
        equation,
        loss_class=PINN_Loss,
        activation=nn.Tanh(),
        scaling_function=None,
        rff_features=20,
        rff_sigma=1.0,
        seed=None,
        num_blocks=3,
        loss_kwargs=None
    ):
        model_kwargs = {"num_blocks" : num_blocks}
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            equation=equation,
            model_class=PirateNet,
            loss_class=loss_class,
            activation=activation,
            scaling_function=scaling_function,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            seed=seed,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs
        )