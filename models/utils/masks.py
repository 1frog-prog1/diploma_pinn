import torch
import torch.nn as nn

class LinearMask(nn.Module):
    """
    Polynomial mask function for scaling the loss weights.
    """
    def __init__(self, c=1.0, q=2):
        super(LinearMask, self).__init__()
        self.c = c
    
    def forward(self, lmbd):
        return self.c * lmbd  # Linear scaling
    
    def backward(self, lmbd):
        return self.c * torch.ones_like(lmbd)  # Gradient is constant


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
    
    def backward(self, lmbd):
        """
        Compute the backward pass of the polynomial mask.
        :param lmbd: Input tensor.
        :return: Gradient of the mask with respect to lmbd.
        """
        return self.c * self.q * lmbd**(self.q - 1)
    
    
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
    
    def backward(self, x):
        """
        Compute the backward pass of the sigmoid mask.
        :param x: Input tensor.
        :return: Gradient of the mask with respect to x.
        """
        return self.c * self.q * torch.exp(-self.q * x) / (1 + torch.exp(-self.q * x))**2