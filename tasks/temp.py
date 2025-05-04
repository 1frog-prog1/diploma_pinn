import torch
import numpy as np

class HarmonicOscillator1D:
    """
    1D Harmonic Oscillator equation class.
    """
    def __init__(self, omega=1.0):
        """
        Initialize the harmonic oscillator with a given angular frequency.
        :param omega: Angular frequency of the oscillator.
        """
        self.omega = omega

    def residual_function(self, model, t, x):
        """
        Compute the residual of the harmonic oscillator equation using the model.
        :param model: The neural network model.
        :param t: Time tensor.
        :return: Residual of the harmonic oscillator equation.
        """
        x, x_t, x_tt = self.compute_derivatives(model, t)
        return x_tt + (self.omega ** 2) * x

    def compute_derivatives(self, model, t):
        """
        Compute the derivatives of the model output with respect to time.
        :param model: The neural network model.
        :param t: Time tensor.
        :return: Tuple of derivatives (x, x_t, x_tt).
        """
        # Compute model predictions
        x = model(t)

        # Compute first derivative (dx/dt)
        x_t = torch.autograd.grad(
            x, t, torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]

        # Compute second derivative (d^2x/dt^2)
        x_tt = torch.autograd.grad(
            x_t, t, torch.ones_like(x_t), create_graph=True, retain_graph=True
        )[0]

        return x, x_t, x_tt

    def initial_conditions(self, t, x0=1.0, v0=0.0):
        """
        Initial conditions for the harmonic oscillator.
        :param t: Time tensor.
        :param x0: Initial position.
        :param v0: Initial velocity.
        :return: Initial condition values.
        """
        if isinstance(t, torch.Tensor):
            x = torch.full_like(t, x0)
            v = torch.full_like(t, v0)
            return x, v
        else:
            raise TypeError("Input must be a torch.Tensor")

    def analytical_solution(self, t, x0=1.0, v0=0.0):
        """
        Analytical solution for the harmonic oscillator.
        :param t: Time tensor or array.
        :param x0: Initial position.
        :param v0: Initial velocity.
        :return: Analytical solution values.
        """
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()

        omega = self.omega
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        return torch.tensor(x, dtype=torch.float32)