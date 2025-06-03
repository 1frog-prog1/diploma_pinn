import torch
import torch.nn as nn
from .base_class import BasePINNLoss       

class DB_PINN_Loss(BasePINNLoss):
    def __init__(self, equation, u_model):
        super(DB_PINN_Loss, self).__init__(equation, u_model)
        self.run_loss_mean = torch.Tensor([])
        self.step = 0

    def forward(self, x_pde, x_ics, x_bcs, x_data=None, u_data=None):
        t_pde, x_pde = x_pde[:, 0:1], x_pde[:, 1:]
        phys_res = self.equation.residual_function(self.u_model, t_pde, x_pde)
        phys_loss = self.compute_physics_loss(phys_res)

        t_bcs = x_bcs[:, 0:1]
        u_bcs = self.equation.boundary_conditions(t_bcs)
        u_pred_bcs = self.u_model(x_bcs)
        bcs_loss = self.compute_boundary_loss(u_bcs, u_pred_bcs)

        t_ics, x_ics_split = x_ics[:, 0:1], x_ics[:, 1:]
        u_ics = self.equation.initial_conditions(x_ics_split)
        u_pred_ics = self.u_model(torch.cat((t_ics, x_ics_split), dim=1))
        ics_loss = self.compute_initial_loss(u_ics, u_pred_ics)

        u_pred_data = self.u_model(x_data) if x_data is not None else None
        data_loss = self.compute_data_loss(u_data, u_pred_data)

        loss_vector = torch.stack(
            [bcs_loss, ics_loss]
        )
        grads_phys = torch.autograd.grad(phys_loss, self.u_model.parameters(), retain_graph=True)
        grads_bcs  = torch.autograd.grad(bcs_loss, self.u_model.parameters(), retain_graph=True)
        grads_ics  = torch.autograd.grad(ics_loss, self.u_model.parameters(), retain_graph=True)
        print(grads_bcs, grads_ics)
        loss_grad_vector = torch.stack([grads_bcs, grads_ics])

        if x_data is None or u_data is None:
            grads_data = torch.autograd.grad(data_loss, self.u_model.parameters(), retain_graph=True)
            loss_vector = torch.stack([loss_vector, data_loss])
            loss_grad_vector = torch.stack([loss_grad_vector, grads_data])

        grad_ratio = torch.sum(
            (grads_phys.abs().max()) / (self.coeff_vector * loss_grad_vector + 1e-8).abs()
        )

        if self.step == 0:
            self.run_loss_mean = loss_vector
        else:
            self.run_loss_mean = (1 - 1 / self.step) * self.run_loss_mean + 1 / self.step * loss_vector
        self.step += 1

        diff_index = loss_vector / self.run_loss_mean
        loss_coeffs = diff_index / diff_index.sum() * grad_ratio

        total_loss = phys_loss + (loss_coeffs * loss_vector).sum()
        return total_loss