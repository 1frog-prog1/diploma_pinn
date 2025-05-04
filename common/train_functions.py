import torch
from models.utils.losses import PINN_Loss

# Освобождение всех данных с CUDA
def move_data_to_device(device, *args):
    updated_args = []
    for arg in args:
        if isinstance(arg, list):
            updated_args.append([item.to(device) for item in arg])  # Перезаписываем элементы списка
        elif arg is not None:
            updated_args.append(arg.to(device))  # Перезаписываем тензор
        else:
            updated_args.append(None)  # Если аргумент None, сохраняем его
    return updated_args

def move_training_data_to_device(
        device, 
        x_data, u_data, x_physics, x_initial, x_boundary,
        model, optimizers
    ):
    x_data, u_data, x_physics, x_initial, x_boundary = move_data_to_device(
        device, x_data, u_data, x_physics, x_initial, x_boundary
    )
    return x_data, u_data, x_physics, x_initial, x_boundary


def train_epoch_pinn(
        model, 
        optimizers, schedulers,
        x_physics, x_initial, x_boundary, 
        x_data=None, u_data=None,
        device="cpu"):
    """
    Обучение PINN за одну эпоху.

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

    # Обнуление градиентов для всех оптимизаторов
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    # Вычисление функции потерь
    loss = model.loss(
        x_pde=x_physics, 
        x_ics=x_initial, 
        x_bcs=x_boundary, 
        x_data=x_data, 
        u_data=u_data
    )
    total_loss += loss.item()

    # Обратный проход и обновление параметров
    loss.backward(retain_graph=True)  # Добавлено retain_graph=True, чтобы предотвратить освобождение графа
    
    # Шаг оптимизации для всех оптимизаторов
    for optimizer in optimizers.values():
        optimizer.step()

    for sched in schedulers.values():
        sched.step()

    return total_loss


def eval_pinn(
        model, x_physics, x_initial, x_boundary, 
        criterion,
        x_data=None, u_data=None, 
        device="cpu"):
    """
    Оценка PINN на базовом лоссе.
    """
    model.eval()
    total_loss = 0.0

    # Вычисление функции потерь
    loss = criterion(
        x_pde=x_physics, 
        x_ics=x_initial, 
        x_bcs=x_boundary, 
        x_data=x_data, 
        u_data=u_data
    )
    total_loss += loss.item()
        
    return total_loss


def train_pinn(
    max_epoch, 
    model, optimizers, schedulers,
    x_physics, x_initial, x_boundary, 
    x_data=None, u_data=None, device="cpu"
):
    """
    Обучение PINN.
    """

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            device, x_data, u_data, x_physics, x_initial,
            x_boundary, model, optimizers
        )
    
    train_losses, eval_losses = [], []
    criterion = PINN_Loss(model.equation, model.u_model)

    for epoch in range(max_epoch):
        # Обучение за одну эпоху
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
            print(f"Epoch: {epoch+1}/{max_epoch}, PINN Loss: {train_loss}, True Loss: {true_loss}")
            # print("Cuda memory_allocated", torch.cuda.memory_allocated())  # Используемая память
            # print("Cuda memory reserved", torch.cuda.memory_reserved()) 
            
        if device == "cuda":
            torch.cuda.empty_cache()

    if device == "cuda":
        x_data, u_data, x_physics, x_initial, x_boundary = move_training_data_to_device(
            "cpu", x_data, u_data, x_physics, x_initial,
            x_boundary, model, optimizers
        )

    return train_losses, eval_losses