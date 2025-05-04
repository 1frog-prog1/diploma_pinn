import torch
import torch.nn as nn

class PolynomialMask(nn.Module):
    """
    Polynomial mask function for scaling the loss weights.
    """
    def __init__(self, c=1.0, q=2):
        """
        Initialize the PolynomialMask with constants c and q.
        :param c: Constant for scaling.
        :param q: Exponent for polynomial scaling.
        """
        super(PolynomialMask, self).__init__()
        self.c = c
        self.q = q

    def forward(self, lmbd):
        """
        Apply the polynomial mask to the input tensor.
        :param lmbd: Input tensor.
        :return: Scaled tensor.
        """
        return self.c * lmbd**self.q  # Polynomial scaling
    
class SigmoidMask(nn.Module):
    """
    Sigmoid mask function for scaling the loss weights.
    """
    def __init__(self, c=1.0, q=4):
        """
        Initialize the SigmoidMask with constants c and q.
        :param c: Constant for scaling.
        :param q: Exponent for sigmoid scaling.
        """
        super(SigmoidMask, self).__init__()
        self.c = c
        self.q = q

    def forward(self, x):
        """
        Apply the sigmoid mask to the input tensor.
        :param x: Input tensor.
        :return: Scaled tensor.
        """
        return self.c / (1 + torch.exp(-self.q * x))  # Sigmoid scaling