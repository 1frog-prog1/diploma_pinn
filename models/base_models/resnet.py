import torch
import torch.nn as nn
from .base_class import BaseModel

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.linear(x))
        return x + out
    

class ResNet(BaseModel):
    def __init__(
        self, input_dim, hidden_layers, output_dim,
        activation=nn.Tanh(), scaling_function=None,
        rff_features=0, rff_sigma=1.0, seed=None
    ):
        super(ResNet, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            scaling_function=scaling_function,
            seed=seed
        )

        in_dim = input_dim + 2 * rff_features if rff_features > 0 else input_dim
        layers = []
        for i in range(len(hidden_layers)):
            layers.append(ResidualBlock(
                in_features=in_dim if i == 0 else hidden_layers[i - 1],
                out_features=hidden_layers[i],
                activation=activation
            ))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess_input(x)
        return self.model(x)