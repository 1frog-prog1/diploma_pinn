import torch
import torch.nn as nn


# PINN Loss
class PINN_Loss(nn.Module):
    def __init__(self, equation, u_model):
        super(PINN_Loss, self).__init__()
        self.equation = equation
        self.u_model = u_model

    def compute_physics_loss(self, phys_res):
        return torch.mean(phys_res ** 2)

    def compute_boundary_loss(self, u_bcs, u_pred_bcs):
        return torch.mean((u_pred_bcs - u_bcs) ** 2)

    def compute_initial_loss(self, u_ics, u_pred_ics):
        return torch.mean((u_pred_ics - u_ics) ** 2)

    def compute_data_loss(self, u_data, u_pred_data):
        if u_pred_data is not None and u_data is not None:
            return torch.mean((u_pred_data - u_data) ** 2)
        return 0.0

    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        # Physics loss
        t_pde, x_pde = x_pde[:, 0:1], x_pde[:, 1:]
        phys_res = self.equation.residual_function(self.u_model, t_pde, x_pde)
        phys_loss = self.compute_physics_loss(phys_res)

        # Boundary conditions loss
        t_bcs = x_bcs[:, 0:1]
        u_bcs = self.equation.boundary_conditions(t_bcs)
        u_pred_bcs = self.u_model(x_bcs)
        bcs_loss = self.compute_boundary_loss(u_bcs, u_pred_bcs)

        # Initial conditions loss
        t_ics, x_ics = x_ics[:, 0:1], x_ics[:, 1:]
        u_ics = self.equation.initial_conditions(x_ics)
        u_pred_ics = self.u_model(torch.cat((t_ics, x_ics), dim=1))
        ics_loss = self.compute_initial_loss(u_ics, u_pred_ics)

        # Data loss
        if x_data is not None and u_data is not None:
            u_pred_data = self.u_model(x_data)
        else:
            u_pred_data = None
        data_loss = self.compute_data_loss(u_data, u_pred_data)
        total_loss = phys_loss + bcs_loss + ics_loss + data_loss
        return total_loss


# SA-PINN Loss Class
class SA_PINN_Loss(nn.Module):
    def __init__(self, points_pde, points_ics, points_bcs, equation, u_model, mask_class):
        super(SA_PINN_Loss, self).__init__()
        self.lambdas_pde = nn.Parameter(torch.ones(points_pde.shape[0]))
        self.lambdas_ics = nn.Parameter(torch.ones(points_ics.shape[0]))
        self.lambdas_bcs = nn.Parameter(torch.ones(points_bcs.shape[0]))
        self.equation = equation
        self.u_model = u_model
        self.mask_class = mask_class
        self.mask = mask_class(c=1.0, q=2)

    def compute_physics_loss(self, res_pred):
        return torch.mean(self.mask(self.lambdas_pde) * (res_pred ** 2))

    def compute_boundary_loss(self, u_bcs, u_pred_bcs):
        return torch.mean(self.mask(self.lambdas_bcs) * ((u_pred_bcs - u_bcs) ** 2))

    def compute_initial_loss(self, u_ics, u_pred_ics):
        return torch.mean(self.mask(self.lambdas_ics) * ((u_pred_ics - u_ics) ** 2))

    def compute_data_loss(self, u_data, u_pred_data):
        if u_pred_data is not None and u_data is not None:
            return torch.mean((u_pred_data - u_data) ** 2)
        return 0.0

    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        # Physics loss
        t_pde, x_pde = x_pde[:, 0:1], x_pde[:, 1:]
        phys_res = self.equation.residual_function(self.u_model, t_pde, x_pde)

        # Boundary conditions loss
        u_bcs = self.equation.boundary_conditions(x_bcs[:, 0:1])
        u_pred_bcs = self.u_model(x_bcs)
        
        u_ics = self.equation.initial_conditions(x_ics[:, 1:])
        u_pred_ics = self.u_model(x_ics)

        loss_data = self.compute_data_loss(x_data, u_data)

        # Total loss using custom autograd function
        total_loss = self.SA_PINN_CustomLoss.apply(
            phys_res, u_pred_bcs, u_bcs, u_pred_ics, u_ics,
            self.lambdas_pde, self.lambdas_bcs, self.lambdas_ics, self.mask_class
        ) + loss_data
        return total_loss

    # Custom autograd function for SA-PINN
    class SA_PINN_CustomLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics, mask_class):
            # Save mask parameters (c and q) instead of the mask object
            ctx.save_for_backward(res_pred, u_pred_bcs, u_bcs, u_pred_ics, u_ics, lambdas_pde, lambdas_bcs, lambdas_ics)
            ctx.mask_class = mask_class

            mask = mask_class(c=1.0, q=2)

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
            mask = mask_class(c=1.0, q=2)

            # Gradients for residuals and predictions
            grad_res_pred = 2 * mask(lambdas_pde) * res_pred / res_pred.numel()
            grad_u_pred_bcs = 2 * mask(lambdas_bcs) * (u_pred_bcs - u_bcs) / u_pred_bcs.numel()
            grad_u_pred_ics = 2 * mask(lambdas_ics) * (u_pred_ics - u_ics) / u_pred_ics.numel()

            # Gradients for lambdas (trainable parameters)
            grad_lambdas_pde = 2 * res_pred ** 2 * mask.c * lambdas_pde ** (mask.q - 1) / res_pred.numel()
            grad_lambdas_bcs = 2 * (u_pred_bcs - u_bcs) ** 2 * mask.c * lambdas_bcs ** (mask.q - 1) / u_pred_bcs.numel()
            grad_lambdas_ics = 2 * (u_pred_ics - u_ics) ** 2 * mask.c * lambdas_ics ** (mask.q - 1) / u_pred_ics.numel()

            # Return gradients for all inputs
            return (
                grad_output * grad_res_pred,  # Gradient for res_pred
                grad_output * grad_u_pred_bcs,  # Gradient for u_pred_bcs
                None,                           # No gradient for u_bcs
                grad_output * grad_u_pred_ics,  # Gradient for u_pred_ics
                None,  # No gradient for u_ics
                -grad_output * grad_lambdas_pde,  # Gradient for lambdas_pde
                -grad_output * grad_lambdas_bcs,  # Gradient for lambdas_bcs
                -grad_output * grad_lambdas_ics,  # Gradient for lambdas_ics
                None  # No gradient for mask
            )
