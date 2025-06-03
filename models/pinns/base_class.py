import torch.nn as nn
from models.base_models import MLP

class BasePINN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        equation,
        loss_class,
        model_class=None,
        activation=nn.Tanh(),
        scaling_function=None,
        rff_features=0,
        rff_sigma=1.0,
        seed=None,
        model_kwargs=None,
        loss_kwargs=None
    ):
        super(BasePINN, self).__init__()
        self.equation = equation

        if model_class is None:
            model_class = MLP
        model_kwargs = model_kwargs or {}
        
        self.u_model = model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            scaling_function=scaling_function,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            seed=seed,
            **model_kwargs
        )

        loss_kwargs = loss_kwargs or {}
        self.loss_fn = loss_class(
            equation=equation,
            u_model=self.u_model,
            **loss_kwargs
        )

    def forward(self, x):
        return self.u_model(x)

    def loss(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        return self.loss_fn(x_pde, x_ics, x_bcs, x_data, u_data)

