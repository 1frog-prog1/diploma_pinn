import numpy as np
import torch
from models.utils.losses import PINN_Loss

# Move all data to the specified device
def move_data_to_device(device, *args):
    updated_args = []
    for arg in args:
        if isinstance(arg, list):
            updated_args.append([item.to(device) for item in arg])  # Update list elements
        elif arg is not None:
            updated_args.append(arg.to(device))  # Update tensor
        else:
            updated_args.append(None)  # Keep None as is
    return updated_args

# Move training data to the specified device
def move_training_data_to_device(
        device, 
        x_data, u_data, x_physics, x_initial, x_boundary
    ):
    x_data, u_data, x_physics, x_initial, x_boundary = move_data_to_device(
        device, x_data, u_data, x_physics, x_initial, x_boundary
    )
    return x_data, u_data, x_physics, x_initial, x_boundary

# Train PINN for one epoch
def train_epoch_pinn(
        model, 
        optimizers, schedulers,
        x_physics, x_initial, x_boundary, 
        x_data=None, u_data=None,
        device="cpu"):
    """
    Train PINN for one epoch.

    Parameters:
    - model: The PINN model.
    - optimizers: Dictionary of optimizers for training.
    - schedulers: Dictionary of learning rate schedulers.
    - x_physics, x_initial, x_boundary: Input tensors for physics, initial, and boundary conditions.
    - x_data, u_data: Optional data for supervised learning.
    - device: Device to run the training on (e.g., 'cpu' or 'cuda').
    """
    model.train()
    total_loss = 0.0

    # Reset gradients for all optimizers
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    # Compute the loss function
    loss = model.loss(
        x_pde=x_physics, 
        x_ics=x_initial, 
        x_bcs=x_boundary, 
        x_data=x_data, 
        u_data=u_data
    )
    total_loss += loss.item()

    # Backward pass and parameter update
    loss.backward(retain_graph=True)  # Added retain_graph=True to prevent graph release
    
    # Optimization step for all optimizers
    for optimizer in optimizers.values():
        optimizer.step()

    for sched in schedulers.values():
        sched.step()

    return total_loss

# Evaluate PINN on the base loss
def eval_pinn(
        model, x_physics, x_initial, x_boundary, 
        criterion,
        x_data=None, u_data=None, 
        device="cpu"):
    """
    Evaluate PINN on the base loss.
    """
    model.eval()
    total_loss = 0.0

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            device, x_data, u_data, x_physics, x_initial, x_boundary
        )

    # Compute the loss function
    loss = criterion(
        x_pde=x_physics, 
        x_ics=x_initial, 
        x_bcs=x_boundary, 
        x_data=x_data, 
        u_data=u_data
    )
    total_loss += loss.item()

    if device == "cuda":
        torch.cuda.empty_cache()

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            "cpu", x_data, u_data, x_physics, x_initial, x_boundary
        )
        
    return total_loss

# Train PINN
def train_pinn(
    max_epoch, 
    model, optimizers, schedulers,
    x_physics, x_initial, x_boundary, 
    x_data=None, u_data=None, device="cpu"
):
    """
    Train PINN.
    """

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            device, x_data, u_data, x_physics, x_initial, x_boundary
        )
    
    train_losses, eval_losses = [], []
    criterion = PINN_Loss(model.equation, model.u_model)

    for epoch in range(max_epoch):
        # Train for one epoch
        train_loss = train_epoch_pinn(
            model, 
            optimizers, schedulers, 
            x_physics, x_initial, x_boundary, 
            x_data, u_data, device
        )
        train_losses.append(train_loss)

        true_loss = eval_pinn(
            model, x_physics, x_initial, x_boundary, 
            criterion, x_data, u_data, device
        )
        eval_losses.append(true_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch+1}/{max_epoch}, PINN Loss: {train_loss:5.5f}, True Loss: {true_loss:5.5f}")
            # print("Cuda memory_allocated", torch.cuda.memory_allocated())  # Used memory
            # print("Cuda memory reserved", torch.cuda.memory_reserved()) 
            
        if device == "cuda":
            torch.cuda.empty_cache()

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            "cpu", x_data, u_data, x_physics, x_initial, x_boundary
        )

    return train_losses, eval_losses


def rad_finetune_pinn(
    max_epoch, 
    model, optimizers, schedulers,
    x_physics, x_initial, x_boundary, 
    x_data=None, u_data=None, device="cpu",
    resample_every_N=2000, resample_percent=0.1,
    k=2, c=1
):
    resample_number_points = int(len(x_physics) * resample_percent)
    criterion = PINN_Loss(model.equation, model.u_model)
    number_of_resample = max_epoch // resample_every_N + 1
    eval_losses = []

    for resample_it in range(number_of_resample):
        eps = model.equation.residual_function(model, x_physics[:, 0:1].to(device), x_physics[:, 1:].to(device)).to('cpu')
        probs = eps**k / torch.mean(eps**k) + c
        probs /= torch.sum(probs)
        probs = probs.detach().numpy().flatten()
        chosen_idxs = np.random.choice(
            len(x_physics), size=resample_number_points, replace=False, p=probs
        )
        x_physics_resampled = x_physics[chosen_idxs]
        finetune_epoch = min(max_epoch - resample_it * resample_every_N, resample_every_N)
        print("Finetuning model on resampled data...")
        train_pinn(
            finetune_epoch, model, optimizers, schedulers, 
            x_physics_resampled, x_initial, x_boundary, x_data, u_data, device
        )

        true_loss = eval_pinn(model, x_physics, x_initial, x_boundary, criterion, x_data, u_data, device)
        eval_losses.append(true_loss)

        print(f"Resample epoch: {resample_it + 1}/{number_of_resample}: Eval Loss: {true_loss:5.5f}")

    return eval_losses