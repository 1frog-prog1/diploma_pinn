import torch
import numpy as np

class BurgersND:
    """
    N-dimensional Burgers equation class.
    """
    def __init__(self, nu=0.01, dim=2):
        """
        Initialize the BurgersND class.
        :param nu: Viscosity coefficient.
        :param dim: Number of spatial dimensions.
        """
        self.nu = nu
        self.dim = dim

    def residual_function(self, model, t, x):
        """
        Compute the residual of the N-dimensional Burgers equation using the model.
        :param model: The neural network model.
        :param t: Time tensor.
        :param x: Spatial tensor (shape: [batch_size, dim]).
        :return: Residual of the Burgers equation.
        """
        u, u_t, u_grad, u_laplacian = self.compute_derivatives(model, t, x)
        u_dot_grad_u = torch.sum(u_grad * u, dim=1, keepdim=True)  # u · ∇u
        return u_t + u_dot_grad_u - self.nu * u_laplacian

    def initial_conditions(self, x):
        """
        Initial conditions for the N-dimensional Burgers equation.
        :param x: Input tensor or array for initial conditions (shape: [batch_size, dim]).
        :return: Initial condition values.
        """
        if isinstance(x, torch.Tensor):
            return -torch.sin(torch.pi * x[:, 0:1])  # Example: sin(πx) in the first dimension
        elif isinstance(x, (np.ndarray, np.number)):
            return -np.sin(np.pi * x[..., 0:1])
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")

    def boundary_conditions(self, t, x):
        """
        Boundary conditions for the N-dimensional Burgers equation.
        :param t: Time tensor.
        :param x: Spatial tensor (shape: [batch_size, dim]).
        :return: Boundary condition values.
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        elif isinstance(t, (np.ndarray, np.number)):
            return np.zeros_like(t)
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")

    def compute_derivatives(self, model, t, x):
        """
        Compute the derivatives of the model output with respect to time and space.
        :param model: The neural network model.
        :param t: Time tensor.
        :param x: Spatial tensor (shape: [batch_size, dim]).
        :return: Tuple of derivatives (u, u_t, ∇u, ∇²u).
        """
        # Compute model predictions
        inputs = torch.cat([t, x], dim=1)  # Concatenate time and space
        u = model(inputs)

        # Compute time derivative
        u_t = torch.autograd.grad(
            u, t, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        # Compute spatial gradient (∇u)
        u_grad = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]

        # Compute spatial Laplacian
        u_laplacian = torch.zeros_like(u)
        for i in range(self.dim):
            u_grad_i = u_grad[:, i:i+1]
            u_laplacian += torch.autograd.grad(
                u_grad_i, x, torch.ones_like(u_grad_i), create_graph=True, retain_graph=True
            )[0][:, i:i+1]

        return u, u_t, u_grad, u_laplacian
