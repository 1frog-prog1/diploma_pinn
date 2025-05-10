import torch
import torch.nn as nn
import numpy as np

def normalize_layer(min_val, max_val):
    """
    Normalizes the input tensor x to the range [min_val, max_val].

    Args:
        x: torch.Tensor of shape (N, d) — N points in d-dimensional space (e.g., [x, t])
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization

    Returns:
        torch.Tensor of shape (N, d) — normalized tensor
    """
    return lambda x: (x - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero


def rff_transform(X, W, b):
    """
    Transforms input data X using Random Fourier Features (RFF).

    Args:
        X: torch.Tensor of shape (N, d) — N points in d-dimensional space (e.g., [x, t])
        W: Matrix of random frequencies (num_features, d)
        b: Vector of random offsets (num_features)

    Returns:
        torch.Tensor of shape (N, 2*num_features)
    """
    projection = X @ W.T + b       # (N, num_features)
    rff = torch.cat([torch.cos(projection), torch.sin(projection)], dim=1)  # (N, 2*num_features)
    rff = rff * (2.0 / W.shape[0]) ** 0.5  # scaling
    return rff


class MLP(nn.Module):
    def __init__(
            self, input_dim, hidden_layers, output_dim, 
            activtion=nn.Tanh(), scaling_function=None,
            rff_features=0, rff_sigma=1.0, seed=None
        ):
        super(MLP, self).__init__()
        self.rff_features = rff_features
        self.rff_sigma = rff_sigma
        self.seed = seed

        if self.rff_features != 0:
            # Generate fixed W and b
            self.W, self.b = self._init_rff(input_dim, rff_sigma, seed)

        layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_dim + 2 * self.rff_features, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(activtion)  # Activation
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self.scaling_function = scaling_function  # Scaling function

    def _init_rff(self, input_dim, rff_sigma, seed):
        """
        Initializes random frequencies W and offsets b for RFF.
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        W = nn.Parameter(
            torch.normal(
                mean=0.0, std=1.0 / rff_sigma, 
                size=(self.rff_features, input_dim), generator=gen
            ),
            requires_grad=False  # W is not trainable
        )
        b = nn.Parameter(
            torch.rand(self.rff_features, generator=gen) * 2 * torch.pi,
            requires_grad=False  # b is not trainable
        )
        return W, b

    def forward(self, x):
        if self.scaling_function is not None:
            x = self.scaling_function(x)  # Apply scaling function
        if self.rff_features != 0:
            x_rff = rff_transform(x, self.W, self.b)
            x = torch.cat([x, x_rff], dim=1)
        return self.model(x)