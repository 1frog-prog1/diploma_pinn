import torch
import torch.nn as nn
from .base_class import BasePINNLoss       

class DB_PINN_Loss(BasePINNLoss):
    def __init__(self, equation, u_model):
        super(DB_PINN_Loss, self).__init__(equation, u_model)
        # self.run_loss_mean = torch.Tensor([])
        self.step = 0
        self.coeff_vector = torch.ones((3, 1))
        self.register_buffer("run_loss_mean", torch.zeros((3, 1)))  # max size

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

        grads_phys = self.__flatten_grad(torch.autograd.grad(phys_loss, self.u_model.parameters(), retain_graph=True))
        grads_bcs  = self.__flatten_grad(torch.autograd.grad(bcs_loss, self.u_model.parameters(), retain_graph=True))
        grads_ics  = self.__flatten_grad(torch.autograd.grad(ics_loss, self.u_model.parameters(), retain_graph=True))

        loss_grad_vector = torch.stack([grads_bcs, grads_ics])
        if x_data is not None and u_data is not None:
            grads_data = self.__flatten_grad(torch.autograd.grad(data_loss, self.u_model.parameters(), retain_graph=True))
            loss_vector = torch.cat([loss_vector, data_loss.unsqueeze(0)], dim=0)
            loss_grad_vector = torch.cat([loss_grad_vector, grads_data.unsqueeze(0)], dim=0)
        
        with torch.no_grad():
            if self.step == 0:
                self.coeff_vector = self.coeff_vector[:loss_vector.shape[0]]
            if self.coeff_vector.device != loss_vector.device:
                self.coeff_vector = self.coeff_vector.to(loss_vector.device)
            grad_ratio = torch.sum(
                (grads_phys.abs().max()) / (self.coeff_vector * loss_grad_vector + 1e-8).abs().mean()
            )
            grad_ratio = grad_ratio.sqrt()

            if self.step == 0:
                self.run_loss_mean = loss_vector
            else:
                self.run_loss_mean = (1 - 1 / self.step) * self.run_loss_mean + 1 / self.step * loss_vector
            self.step += 1
            diff_index = loss_vector / self.run_loss_mean
            loss_coeffs = diff_index / diff_index.sum() * grad_ratio
            loss_coeffs = loss_coeffs.unsqueeze(1)
            self.coeff_vector = (1 - 1 / self.step) * self.coeff_vector + 1 / self.step * loss_coeffs

        total_loss = phys_loss + (self.coeff_vector * loss_vector).sum()
        return total_loss
    
    def __flatten_grad(self, grads):
        return torch.cat([g.view(-1) for g in grads if g is not None])