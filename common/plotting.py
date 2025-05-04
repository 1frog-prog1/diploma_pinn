import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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

def plot_loss(eval_losses, title="Loss Function", save_path=None):
    """
    Plots the loss function on a semilogarithmic scale.

    :param eval_losses: List or array of loss values.
    :param title: Title for the plot.
    :param save_path: Path to save the figure (optional).
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(eval_losses)), eval_losses, 'k-', label='Loss')
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

