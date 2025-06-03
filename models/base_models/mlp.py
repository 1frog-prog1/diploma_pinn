import torch
import torch.nn as nn
from .base_class import BaseModel

class MLP(BaseModel):
    def __init__(
        self, input_dim, hidden_layers, output_dim,
        activation=nn.Tanh(), scaling_function=None,
        rff_features=0, rff_sigma=1.0, seed=None
    ):
        super(MLP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            scaling_function=scaling_function,
            seed=seed
        )

        in_dim = input_dim + 2 * rff_features if rff_features > 0 else input_dim
        layers = []
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_layers[i - 1], h))
            layers.append(activation)
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess_input(x)
        return self.model(x)
