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
    


    # Лосс-функция
    def loss(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        """
        Вычисляет функцию потерь для SA-PINN.

        Параметры:
        - x_data: входные данные для обучения (тензор).
        - u_data: истинные значения для данных (тензор).
        - x_pde: входные данные для физического уравнения (тензор).
        - x_ics: входные данные для начальных условий (тензор).
        - x_bcs: входные данные для граничных условий (тензор).

        Возвращает:
        - Общая потеря (сумма data loss и physics loss).
        """
        return self.loss_class(x_pde, x_ics, x_bcs, x_data, u_data)
