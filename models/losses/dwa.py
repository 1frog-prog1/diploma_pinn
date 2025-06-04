import torch
import torch.nn as nn
from .base_class import BasePINNLoss   
from models.utils.masks import LinearMask

class DWA_Loss(BasePINNLoss):
    def __init__(self, equation, u_model, mask_class=LinearMask):
        super(DWA_Loss, self).__init__(equation, u_model)
        self.lambda_pde = nn.Parameter(torch.ones((1)))
        self.lambda_ics = nn.Parameter(torch.ones((1)))
        self.lambda_bcs = nn.Parameter(torch.ones((1)))
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

        total_loss = self.DWA_CustomLoss.apply(
            phys_res, u_pred_bcs, u_bcs, u_pred_ics, u_ics,
            self.lambda_pde, self.lambda_bcs, self.lambda_ics, self.mask_class
        ) + loss_data

        # print(f"Losses: PDE: {self.lambda_pde.item()}, BCs: {self.lambda_bcs.item()}, ICs: {self.lambda_ics.item()}")

        return total_loss
    class DWA_CustomLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambda_pde, lambda_bcs, lambda_ics, mask_class):
            ctx.save_for_backward(res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambda_pde, lambda_bcs, lambda_ics)
            ctx.mask_class = mask_class

            mask = mask_class()
            loss_pde = torch.mean(mask(lambda_pde) * res_pred ** 2)
            loss_bcs = torch.mean(mask(lambda_bcs) * (u_pred_bcs - u_bcs) ** 2)
            loss_ics = torch.mean(mask(lambda_ics) * (u_pred_ics - u_ics) ** 2)

            return loss_pde + loss_bcs + loss_ics

        @staticmethod
        def backward(ctx, grad_output):
            res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambda_pde, lambda_bcs, lambda_ics = ctx.saved_tensors
            mask = ctx.mask_class()

            grad_res_pred = 2 * mask(lambda_pde) * res_pred / res_pred.numel()
            grad_u_pred_bcs = 2 * mask(lambda_bcs) * (u_pred_bcs - u_bcs) / u_pred_bcs.numel()
            grad_u_pred_ics = 2 * mask(lambda_ics) * (u_pred_ics - u_ics) / u_pred_ics.numel()

            grad_lambda_pde = mask.backward(lambda_pde) * res_pred ** 2 / res_pred.numel()
            grad_lambda_bcs = mask.backward(lambda_bcs) * (u_pred_bcs - u_bcs) ** 2 / u_pred_bcs.numel()
            grad_lambda_ics = mask.backward(lambda_ics) * (u_pred_ics - u_ics) ** 2 / u_pred_ics.numel()

            return (
                grad_output * grad_res_pred,
                grad_output * grad_u_pred_bcs,
                None,
                grad_output * grad_u_pred_ics,
                None,
                -grad_output * grad_lambda_pde,
                -grad_output * grad_lambda_bcs,
                -grad_output * grad_lambda_ics,
                None
            )