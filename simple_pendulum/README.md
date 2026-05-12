# 1D Simple Pendulum: Standard PINNs vs. SIREN

## Problem overview

This section of the project focuses on solving the simple pendulum, a classic non-linear dynamical system. This problem serves as a fundamental benchmark to evaluate how Physics-Informed Neural Networks (PINNs) learn periodic trajectories governed by differential equations.

We describe the trajectory of the mass using $\theta(t)$, which corresponds to the angle between the pendulum and the vertical axis. The motion is governed by the following second-order non-linear ordinary differential equation (ODE): 

$$
\begin{equation*}
\ddot{\theta} + \omega^2\sin\theta = 0
\end{equation*}
$$

where $\omega = \sqrt{g/L}$ is the angular frequency. The system is subject to the following initial conditions (released from rest at an initial angle):

$$
\begin{equation*}
\theta(0) = \theta_0 \quad , \quad \dot{\theta}(0)=0
\end{equation*}
$$

## Neural Network Architectures

To solve this ODE and predict the trajectory $\theta(t)$ over time, we implemented and compared two different neural network architectures from scratch using **PyTorch**:

1. **Standard FCNN (Fully Connected Neural Network):** A classical data-driven architecture using standard activation functions (like Tanh). While it acts as a good baseline, standard FCNNs suffer from *spectral bias*, meaning they struggle to learn high-frequency periodic functions and their higher-order derivatives efficiently over long time domains.
   
2. **SIREN (Sinusoidal Representation Network):**
   To overcome the limitations of the standard FCNN, we incorporated a SIREN architecture. Unlike traditional networks, SIREN uses periodic sine functions as activation functions. This mathematical property allows the network to capture complex, high-frequency oscillations and perfectly model the first and second derivatives ($\dot{\theta}$ and $\ddot{\theta}$) required by the physical loss function.

## Training Strategy
The models are trained using a composite loss function that combines:
- **Data Loss:** Mean Squared Error (MSE) comparing the network's prediction with a small set of exact analytical data points (calculated using Jacobi elliptic functions for large amplitudes) during the initial training phase.
- **Physics Loss:** The residual of the ODE $\left(\ddot{\theta}_{pred} + \omega^2\sin\theta_{pred}\right)$ evaluated at a dense set of physical collocation points across the entire time domain.

## Acknowledgments
- Credits to **Vincent Sitzmann** (`vsitzmann` on GitHub) for the formulation and original implementation ideas of the SIREN neural network architecture.
- Credits to **Ben Moseley** for the original implementation of the standard FCNN.