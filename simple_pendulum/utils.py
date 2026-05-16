"""
    Module containing utility functions for mathematical and physical computations,
    independent of the neural network architectures.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import ellipj

def get_analytical_pendulum(t, w, theta0):
    """
    Computes the exact analytical solution for the simple pendulum using
    Jacobi elliptic functions.

    Args:
        t (torch.Tensor or numpy.ndarray): Vector of times at which to evaluate the solution.
        w (float): Angular frequency of the pendulum.
        theta0 (float): Initial angle (amplitude) in radians.

    Returns:
        torch.Tensor: Vector with the exact values of the angle (theta).
    """
    k = np.sin(theta0 / 2)
    if torch.is_tensor(t):
        t_np = t.detach().cpu().numpy()
    else:
        t_np = t
        
    u = w * (t_np + (np.pi / (2 * w)))
    sn, cn, dn, ph = ellipj(u, k**2)
    theta_np = 2 * np.arcsin(k * sn)
    return torch.tensor(theta_np, dtype=torch.float32)


def plot_results(results_list, global_title, filename):
    """
    Plots the results of the pendulum experiments, showing the exact solution,
    the model predictions, and the training data points for different numbers of oscillations. 

    Args:
        results_list (list of dict): List of dictionaries, each containing the results 
                                     for a specific number of oscillations. Each dictionary 
                                     should have the keys:
            - 't': Time points for evaluation.
            - 'theta_exact': Exact solution values at the time points.
            - 'theta_pred': Model predictions at the time points.
            - 't_trn': Time points of the training data.
            - 'theta_trn': Training data values at the training time points.
            - 'n_osc': Number of oscillations corresponding to the results.
        global_title (str): Title for the entire figure.
        filename (str): Name of the file to save the plot (should include .png extension).
    """
    fig, axs = plt.subplots(len(results_list), 1, figsize=(10, 3 * len(results_list)), sharex=True)
    if len(results_list) == 1:
        axs = [axs]
        
    fig.suptitle(global_title)
    
    for idx, res in enumerate(results_list):
        ax = axs[idx]
        t = res['t']
        t_trn = res['t_trn']
        
        ax.plot(t, res['theta_exact'], label="Analytical Solution", color='blue', alpha=0.3, lw=2)
        ax.plot(t, res['theta_pred'], label="Prediction", color='blue', linestyle='--')
        ax.scatter(t_trn, res['theta_trn'], color='red', s=30, label="Training Data", zorder=5)
        
        t_max_trn = t_trn.max()
        ax.axvspan(0, t_max_trn, color='grey', alpha=0.15, label='Training Zone')
        ax.axvspan(t_max_trn, t.max(), color='orange', alpha=0.05, label='Test Zone')
        
        ax.set_title(f"$n_{{\\text{{osc}}}} = {res['n_osc']}$")
        ax.set_ylabel(f"$\\theta$ (rad)")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right')
            
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    filepath = os.path.join('plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"-> Plot saved to: {filepath}")

    plt.show()