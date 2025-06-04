import torch
import torch.nn as nn
from .base_class import BasePINNLoss

class SA_PINN_Loss(BasePINNLoss):
    def __init__(self, points_pde, points_ics, points_bcs, equation, u_model, mask_class):
        super(SA_PINN_Loss, self).__init__(equation, u_model)
        self.lambdas_pde = nn.Parameter(torch.ones((points_pde.shape[0], 1)))
        self.lambdas_ics = nn.Parameter(torch.ones((points_ics.shape[0], 1)))
        self.lambdas_bcs = nn.Parameter(torch.ones((points_bcs.shape[0], 1)))
        self.mask_class = mask_class
        self.mask = mask_class()

    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        t_pde, x_pde_split = x_pde[:, 0:1], x_pde[:, 1:]
        phys_res = self.equation.residual_function(self.u_model, t_pde, x_pde_split)

        u_bcs = self.equation.boundary_conditions(x_bcs[:, 0:1])
        u_pred_bcs = self.u_model(x_bcs)

        u_ics = self.equation.initial_conditions(x_ics[:, 1:])
        u_pred_ics = self.u_model(x_ics)

        loss_data = self.compute_data_loss(x_data, u_data)

        total_loss = self.SA_PINN_CustomLoss.apply(
            phys_res, u_pred_bcs, u_bcs, u_pred_ics, u_ics,
            self.lambdas_pde, self.lambdas_bcs, self.lambdas_ics, self.mask_class
        ) + loss_data

        return total_loss

    class SA_PINN_CustomLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics, mask_class):
            ctx.save_for_backward(res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics)
            ctx.mask_class = mask_class

            mask = mask_class()
            loss_pde = torch.mean(mask(lambdas_pde) * res_pred ** 2)
            loss_bcs = torch.mean(mask(lambdas_bcs) * (u_pred_bcs - u_bcs) ** 2)
            loss_ics = torch.mean(mask(lambdas_ics) * (u_pred_ics - u_ics) ** 2)

            return loss_pde + loss_bcs + loss_ics

        @staticmethod
        def backward(ctx, grad_output):
            res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics = ctx.saved_tensors
            mask = ctx.mask_class()

            grad_res_pred = 2 * mask(lambdas_pde) * res_pred / res_pred.numel()
            grad_u_pred_bcs = 2 * mask(lambdas_bcs) * (u_pred_bcs - u_bcs) / u_pred_bcs.numel()
            grad_u_pred_ics = 2 * mask(lambdas_ics) * (u_pred_ics - u_ics) / u_pred_ics.numel()

            grad_lambdas_pde = mask.backward(lambdas_pde) * res_pred ** 2 / res_pred.numel()
            grad_lambdas_bcs = mask.backward(lambdas_bcs) * (u_pred_bcs - u_bcs) ** 2 / u_pred_bcs.numel()
            grad_lambdas_ics = mask.backward(lambdas_ics) * (u_pred_ics - u_ics) ** 2 / u_pred_ics.numel()

            return (
                grad_output * grad_res_pred,
                grad_output * grad_u_pred_bcs,
                None,
                grad_output * grad_u_pred_ics,
                None,
                -grad_output * grad_lambdas_pde,
                -grad_output * grad_lambdas_bcs,
                -grad_output * grad_lambdas_ics,
                None
            )

        @staticmethod
        def forward(ctx, res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics, mask_class):
            # Save mask parameters (c and q) instead of the mask object
            ctx.save_for_backward(res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics)
            ctx.mask_class = mask_class

            mask = mask_class()

            # Physics loss
            loss_pde = torch.mean(mask(lambdas_pde) * (res_pred ** 2))

            # Boundary conditions loss
            loss_bcs = torch.mean(mask(lambdas_bcs) * ((u_pred_bcs - u_bcs) ** 2))

            # Initial conditions loss
            loss_ics = torch.mean(mask(lambdas_ics) * ((u_pred_ics - u_ics) ** 2))

            total_loss = loss_pde + loss_bcs + loss_ics
            return total_loss

        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve saved tensors and mask parameters
            res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics = ctx.saved_tensors
            mask_class = ctx.mask_class
            mask = mask_class()

            # Gradients for residuals and predictions
            grad_res_pred = 2 * mask(lambdas_pde) * (res_pred) / res_pred.numel()
            grad_u_pred_bcs = 2 * mask(lambdas_bcs) * (u_pred_bcs - u_bcs) / u_pred_bcs.numel()
            grad_u_pred_ics = 2 * mask(lambdas_ics) * (u_pred_ics - u_ics) / u_pred_ics.numel()

            # Gradients for lambdas (trainable parameters)
            grad_lambdas_pde = mask.backward(lambdas_pde) * res_pred ** 2  / res_pred.numel()
            grad_lambdas_bcs = mask.backward(lambdas_bcs) * (u_pred_bcs - u_bcs) ** 2 / u_pred_bcs.numel()
            grad_lambdas_ics = mask.backward(lambdas_ics) * (u_pred_ics - u_ics) ** 2 / u_pred_ics.numel()

            # Return gradients for all inputs
            return (
                grad_output * grad_res_pred,  # Gradient for res_pred
                grad_output * grad_u_pred_bcs,  # Gradient for u_pred_bcs
                None,                           # No gradient for u_bcs
                grad_output * grad_u_pred_ics,  # Gradient for u_pred_ics
                None,  # No gradient for u_ics
                -grad_output * grad_lambdas_pde,  # Antigradient for lambdas_pde
                -grad_output * grad_lambdas_bcs,  # Antigradient for lambdas_bcs
                -grad_output * grad_lambdas_ics,  # Antigradient for lambdas_ics
                None  # No gradient for mask
            )
        