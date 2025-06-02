from torch import nn
from .utils.base_classes import BasePINN
from .utils.losses import SA_PINN_Loss
from .utils.masks import PolynomialMask

class SA_PINN(BasePINN):
    def __init__(
        self,
        input_dim, output_dim, hidden_layers,
        equation,
        points_pde=None, points_ics=None, points_bcs=None,
        mask_class=PolynomialMask,
        activation=nn.Tanh(), scaling_function=None,
        rff_features=0, rff_sigma=1.0, seed=None
    ):
        loss_kwargs = {
            "points_pde": points_pde,
            "points_ics": points_ics,
            "points_bcs": points_bcs,
            "mask_class": mask_class
        }
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            equation=equation,
            loss_class=SA_PINN_Loss,
            activation=activation,
            scaling_function=scaling_function,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            seed=seed,
            loss_kwargs=loss_kwargs
        )

    @property
    def lambdas(self):
        return [self.loss_fn.lambdas_pde,
                self.loss_fn.lambdas_ics,
                self.loss_fn.lambdas_bcs]

    @property
    def lambdas_pde(self):
        return self.loss_fn.lambdas_pde

    @property
    def lambdas_ics(self):
        return self.loss_fn.lambdas_ics

    @property
    def lambdas_bcs(self):
        return self.loss_fn.lambdas_bcs
