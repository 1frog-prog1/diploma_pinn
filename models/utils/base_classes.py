import torch
import torch.nn as nn
import torch.autograd as autograd
from .mlp import MLP
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


class BasePINNLoss(nn.Module, ABC):
    def __init__(self, equation, u_model):
        super(BasePINNLoss, self).__init__()
        self.equation = equation
        self.u_model = u_model

    def compute_physics_loss(self, residual):
        return torch.mean(residual ** 2)

    def compute_boundary_loss(self, u_bcs, u_pred_bcs):
        return torch.mean((u_pred_bcs - u_bcs) ** 2)

    def compute_initial_loss(self, u_ics, u_pred_ics):
        return torch.mean((u_pred_ics - u_ics) ** 2)

    def compute_data_loss(self, u_data, u_pred_data):
        if u_pred_data is not None and u_data is not None:
            return torch.mean((u_pred_data - u_data) ** 2)
        return torch.tensor(0.0, device=self.get_model_device())
    
    def get_model_device(self):
        return next(self.u_model.parameters()).device

    @abstractmethod
    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        pass