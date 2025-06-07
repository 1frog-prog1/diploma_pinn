import torch
import numpy as np
from scipy.integrate import quad

class Burgers1D:
    """
    1D Burgers equation class.
    """
    def __init__(self, nu=0.01):
        self.nu = nu

    def residual_function(self, model, t, x):
        """
        Compute the residual of the Burgers equation using the model.
        :param model: The neural network model.
        :param t: Time tensor.
        :param x: Spatial tensor.
        :return: Residual of the Burgers equation.
        """
        u, u_t, u_x, u_xx = self.compute_derivatives(model, t, x)
        return torch.abs(u_t + u * u_x - self.nu * u_xx)
    
    def initial_conditions(self, x):
        """
        Initial conditions for the Burgers equation.
        :param x: Input tensor or array for initial conditions.
        :return: Initial condition values.
        """
        if isinstance(x, torch.Tensor):
            return -torch.sin(torch.pi * x)
        elif isinstance(x, (np.ndarray, np.number)):
            return -np.sin(np.pi * x)
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")
    
    def boundary_conditions(self, t):
        """
        Boundary conditions for the Burgers equation.
        :param t: Time tensor or array for boundary conditions.
        :return: Boundary condition values.
        """
        if isinstance(t, torch.Tensor):
            return torch.zeros_like(t)
        elif isinstance(t, (np.ndarray, np.number)):
            return np.zeros_like(t)
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")
    
    def analytical_solution(self, t, x):  # плохо считается при малых nu
        """
        Analytical solution for the Burgers equation using Cole-Hopf transformation.
        :param t: 2D array of time values (from np.meshgrid).
        :param x: 2D array of spatial values (from np.meshgrid).
        :return: 2D array of solution values u(t, x).
        """
        def phi0(y):
            """
            Initial condition for φ in Cole-Hopf transformation.
            :param y: Spatial coordinate.
            :return: Value of φ0.
            """
            return np.exp(-1 / (2 * self.nu * np.pi) * np.cos(np.pi * y))

        def G(x_minus_y, t):
            """
            Green's function for the heat equation.
            :param x_minus_y: Difference between x and y.
            :param t: Time value.
            :return: Value of Green's function.
            """
            eps = 1e-10  # Small constant to avoid division by zero
            return (1 / np.sqrt(4 * np.pi * self.nu * (t + eps))) * np.exp(-(x_minus_y**2) / (4 * self.nu * (t + eps)))

        def phi(x_val, t_val):
            """
            Compute φ(x, t) using the Cole-Hopf transformation.
            :param x_val: Spatial coordinate.
            :param t_val: Time value.
            :return: Value of φ(x, t).
            """
            if t_val <= 0:
                return phi0(x_val)
            integrand = lambda y: G(x_val - y, t_val) * phi0(y)
            result, _ = quad(integrand, -1, 1)
            return result

        def compute_u(t_vals, x_vals):
            """
            Compute u(x, t) for arrays of x and t values.
            :param x_vals: 2D array of spatial coordinates.
            :param t_vals: 2D array of time coordinates.
            :return: 2D array of solution values u(t, x).
            """
            u_vals = np.zeros_like(x_vals)
            for i in range(x_vals.shape[0]):
                for j in range(x_vals.shape[1]):
                    x_val = x_vals[i, j]
                    t_val = t_vals[i, j]
                    # Compute φ and its spatial derivative
                    dx = 1e-13
                    phi_val = phi(x_val, t_val)
                    phi_plus = phi(x_val + dx, t_val)
                    phi_minus = phi(x_val - dx, t_val)
                    if phi_val <= 0 or phi_plus <= 0 or phi_minus <= 0:
                        u_vals[i, j] = 0  # Handle invalid values
                    elif x_val == x_vals[0, 0] or x_val == x_vals[-1, -1]:
                        u_vals[i, j] = self.boundary_conditions(t_val)
                    else:
                        dln_phi_dx = (np.log(phi_plus) - np.log(phi_minus)) / (2 * dx)
                        # Compute u using Cole-Hopf transformation
                        u_vals[i, j] = -2 * self.nu * dln_phi_dx
            return u_vals

        u_vals = compute_u(t, x)
        u_vals[:, 0] = self.boundary_conditions(t[:, 0])
        u_vals[:, -1] = self.boundary_conditions(t[:, -1])
        return u_vals
    
    # -----------------------------
    def compute_derivatives(self, model, t, x):
        """
        Compute the derivatives of the model output with respect to time and space.
        :param model: The neural network model.
        :param t: Time tensor.
        :param x: Spatial tensor.
        :return: Tuple of derivatives (u, u_t, u_x, u_xx).
        """
        # Compute model predictions
        u = model(torch.cat([t, x], dim=1))

        # Compute derivatives
        u_t = torch.autograd.grad(
            u, t, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
        )[0]

        return u, u_t, u_x, u_xx