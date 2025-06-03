import torch
import torch.nn as nn
from .base_class import BasePINNLoss

class PINN_Loss(BasePINNLoss):
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

        return phys_loss + bcs_loss + ics_loss + data_loss
