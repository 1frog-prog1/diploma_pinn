import torch
import torch.nn as nn
from .base_classes import BaseModel  # Предполагается, что base_nn уже реализован

class PirateNetBlock(nn.Module):
    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.activation = activation

    def forward(self, x, U, V):
        f = self.activation(self.W1(x))
        z1 = f * U + (1 - f) * V

        g = self.activation(self.W2(z1))
        z2 = g * U + (1 - g) * V

        h = self.activation(self.W3(z2))
        return self.alpha * h + (1 - self.alpha) * x


class PirateNet(BaseModel):
    def __init__(
        self, input_dim, hidden_dim, output_dim,
        num_blocks=3,
        activation=nn.Tanh(), scaling_function=None,
        rff_features=0, rff_sigma=1.0, seed=None
    ):
        super(PirateNet, self).__init__(
            input_dim=input_dim,
            rff_features=rff_features,
            rff_sigma=rff_sigma,
            scaling_function=scaling_function,
            seed=seed
        )

        in_dim = input_dim + 2 * rff_features if rff_features > 0 else input_dim
        self.activation = activation

        self.gate_U = nn.Linear(in_dim, hidden_dim)
        self.gate_V = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            PirateNetBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])

        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.preprocess_input(x)
        U = self.activation(self.gate_U(x))
        V = self.activation(self.gate_V(x))
        for block in self.blocks:
            x = block(x, U, V)
        return self.final_layer(x)
