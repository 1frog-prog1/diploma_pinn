import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import torch

# ========================= 2D Plotting Functions =========================

def compute_approximation_2d(model, lb, ub, N=50):
    """
    Compute approximate values of a function using a PyTorch model for 2D data.

    :param model: PyTorch model to evaluate.
    :param lb: Lower bounds as a tensor (e.g., [tmin, xmin]).
    :param ub: Upper bounds as a tensor (e.g., [tmax, xmax]).
    :param N: Number of points for spatial discretization.
    :return: Meshgrid (T, X) and reshaped predictions (U).
    """
    # Set up meshgrid
    tspace = np.linspace(lb[0].item(), ub[0].item(), N + 1)
    xspace = np.linspace(lb[1].item(), ub[1].item(), N + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = torch.tensor(np.vstack([T.flatten(), X.flatten()]).T, dtype=torch.float32)

    # Determine predictions of u(t, x)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        upred = model(Xgrid)

    # Reshape predictions
    U = upred.numpy().reshape(N + 1, N + 1)
    return T, X, U

def plot_heatmaps_2d(tspace, xspace, datasets, titles, cmap='seismic', save_path=None):
    """
    Plots multiple heatmaps on a grid for 2D data.

    :param tspace: Time grid values.
    :param xspace: Space grid values.
    :param datasets: List of 2D arrays to plot.
    :param titles: List of titles for each subplot.
    :param cmap: Colormap for the heatmaps.
    :param save_path: Path to save the figure (optional).
    """
    # Ensure tspace and xspace are NumPy arrays
    tspace = np.array(tspace) if isinstance(tspace, list) else tspace
    xspace = np.array(xspace) if isinstance(xspace, list) else xspace

    fig, ax = plt.subplots(ncols=len(datasets), figsize=(12, 6), sharey=True)
    for i, (title, data) in enumerate(zip(titles, datasets)):
        ax[i].set_title(title, fontsize=14)
        ax[i].set_xlabel('$t$', fontsize=14)
        ax[i].set_ylabel('$x$', fontsize=14)
        im = ax[i].imshow(
            data, extent=[tspace.min(), tspace.max(), xspace.min(), xspace.max()],
            origin='lower', aspect='auto', cmap=cmap
        )
        fig.colorbar(im, ax=ax[i], label='$u_\\theta(t,x)$')
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_single_heatmap_2d(tspace, xspace, data, title, cmap='seismic', save_path=None):
    """
    Plots a single heatmap for 2D data.

    :param tspace: Time grid values.
    :param xspace: Space grid values.
    :param data: 2D array to plot.
    :param title: Title for the plot.
    :param cmap: Colormap for the heatmap.
    :param save_path: Path to save the figure (optional).
    """
    # Ensure tspace and xspace are NumPy arrays
    tspace = np.array(tspace) if isinstance(tspace, list) else tspace
    xspace = np.array(xspace) if isinstance(xspace, list) else xspace

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$t$', fontsize=14)
    ax.set_ylabel('$x$', fontsize=14)
    im = ax.imshow(
        data, extent=[tspace.min(), tspace.max(), xspace.min(), xspace.max()],
        origin='lower', aspect='auto', cmap=cmap
    )
    fig.colorbar(im, ax=ax, label='$u_\\theta(t,x)$')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format x-axis as float with 2 decimals
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))  # Format y-axis as float with 2 decimals
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)


def plot_loss(eval_losses, title="Loss Function", save_path=None, alpha=1.0):
    """
    Plots the loss function(s) on a semilogarithmic scale.

    :param eval_losses: List or array of loss values, a list of lists/arrays
                        of loss values for multiple curves, or a dictionary
                        where keys are labels and values are lists/arrays of losses.
    :param title: Title for the plot.
    :param save_path: Path to save the figure (optional).
    :param alpha: Transparency level for the plotted lines (0.0 to 1.0).
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    all_losses = []

    if isinstance(eval_losses, dict):
        # Plot multiple curves from a dictionary
        for label, losses in eval_losses.items():
            if isinstance(losses, (list, np.ndarray)):
                ax.semilogy(range(len(losses)), losses, label=label, alpha=alpha)
                all_losses.extend(losses)
            else:
                print(f"Warning: Value for key '{label}' is not a list or array and will be skipped.")

    elif isinstance(eval_losses, list) and all(isinstance(l, (list, np.ndarray)) for l in eval_losses):
        # Plot multiple curves from a list of lists/arrays
        for i, losses in enumerate(eval_losses):
            ax.semilogy(range(len(losses)), losses, label=f'Loss {i+1}', alpha=alpha)
            all_losses.extend(losses)

    elif isinstance(eval_losses, (list, np.ndarray)):
        # Plot a single curve (original functionality)
        ax.semilogy(range(len(eval_losses)), eval_losses, 'k-', label='Loss', alpha=alpha)
        all_losses.extend(eval_losses)
    else:
        print("Error: Input 'eval_losses' must be a list, numpy array, list of lists/arrays, or a dictionary.")
        plt.close(fig) # Close the figure if input is invalid
        return

    if all_losses:
        min_loss = min(all_losses)
        max_loss = max(all_losses)

    # Определяем диапазон степеней 10
    # Используем floor для нижней границы и ceil для верхней
    start_power = math.floor(math.log10(min_loss))
    end_power = math.ceil(math.log10(max_loss))

    # Генерируем метки оси Y как степени 10
    y_ticks = [10**p for p in range(int(start_power), int(end_power) + 1)]

    # Устанавливаем метки оси Y
    ax.set_yticks(y_ticks)
    # Форматируем метки как 10 в степени (используем LaTeX для красивого отображения)
    ax.set_yticklabels([f'$10^{{{p}}}$' for p in range(int(start_power), int(end_power) + 1)])

    ax.set_xlabel('$n_{epoch}$', fontsize=14)
    ax.set_ylabel('$\mathcal{L}$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(fontsize=12)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

# ========================= Other Dimensional Functions =========================

# Add other dimensional plotting or computation functions here as needed.

