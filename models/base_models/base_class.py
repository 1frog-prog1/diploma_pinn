import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers=None,
        activation=nn.Tanh(),
        scaling_function=None,
        rff_features=0,
        rff_sigma=1.0,
        seed=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or []
        self.activation = activation
        self.scaling_function = scaling_function
        self.rff_features = rff_features
        self.rff_sigma = rff_sigma
        self.seed = seed

        if self.rff_features > 0:
            self.W, self.b = self._init_rff()
        else:
            self.W, self.b = None, None

    def _init_rff(self):
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed)

        W = torch.normal(
            mean=0.0,
            std=1.0 / self.rff_sigma,
            size=(self.rff_features, self.input_dim),
            generator=gen
        )
        self.register_buffer("W", W)

        b = torch.rand(self.rff_features, generator=gen) * 2 * torch.pi
        self.register_buffer("b", b)
        return W, b

    def rff_transform(self, x):
        projection = x @ self.W.T + self.b
        rff = torch.cat([torch.cos(projection), torch.sin(projection)], dim=1)
        return rff * (2.0 / self.W.shape[0]) ** 0.5

    def preprocess_input(self, x):
        if self.scaling_function:
            x = self.scaling_function(x)
        if self.rff_features > 0:
            x_rff = self.rff_transform(x)
            x = torch.cat([x, x_rff], dim=1)
        return x

    @abstractmethod
    def forward(self, x):
        pass