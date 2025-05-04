import torch
import torch.nn as nn
import numpy as np

def rff_transform(X, W, b):
    """
    Преобразует входные данные X с помощью Random Fourier Features (RFF).

    Аргументы:
        X: torch.Tensor формы (N, d) — N точек в d-мерном пространстве (например, [x, t])
        W: Матрица случайных частот (num_features, d)
        b: Вектор случайных смещений (num_features)

    Возвращает:
        torch.Tensor формы (N, 2*num_features)
    """
    projection = X @ W.T + b       # (N, num_features)
    rff = torch.cat([torch.cos(projection), torch.sin(projection)], dim=1)  # (N, 2*num_features)
    rff = rff * (2.0 / W.shape[0]) ** 0.5  # масштабирование
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
            # Генерация фиксированных W и b
            self.W, self.b = self._init_rff(input_dim, rff_sigma, seed)

        layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_dim + 2 * self.rff_features, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(activtion)  # Активация
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self.scaling_function = scaling_function  # Функция масштабирования

    def _init_rff(self, input_dim, rff_sigma, seed):
        """
        Инициализация случайных частот W и смещений b для RFF.
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        W = nn.Parameter(
            torch.normal(
                mean=0.0, std=1.0 / rff_sigma, 
                size=(self.rff_features, input_dim), generator=gen
            ),
            requires_grad=False  # W не обучается
        )
        b = nn.Parameter(
            torch.rand(self.rff_features, generator=gen) * 2 * torch.pi,
            requires_grad=False  # b не обучается
        )
        return W, b

    def forward(self, x):
        if self.scaling_function is not None:
            x = self.scaling_function(x)  # Apply scaling function
        if self.rff_features != 0:
            x_rff = rff_transform(x, self.W, self.b)
            x = torch.cat([x, x_rff], dim=1)
        return self.model(x)