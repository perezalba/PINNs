"""
    Module containing utility functions for mathematical and physical computations,
    independent of the neural network architectures.
"""
import torch
import numpy as np
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