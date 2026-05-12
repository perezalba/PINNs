# 1D Simple Pendulum: Standard PINNs vs. SIREN

## Problem overview

This section of the project focuses on solving the simple pendulum, a classic non-linear dynamical system. This problem serves as a fundamental benchmark to evaluate how Physics-Informed Neural Networks (PINNs) learn periodic trajectories governed by differential equations.

We describe the trajectory of the mass using $\theta(t)$, which corresponds to the angle between the pendulum and the vertical axis. The motion is governed by the following ordinary differential equation (ODE): 

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

To solve this ODE and predict the trajectory $\theta(t)$ over time, we implemented and compared three distinct neural network configurations from scratch using **PyTorch**:

1. **Purely Data-Driven FCNN (Standard FCNN):** A classical neural network architecture using standard activation functions (like Tanh) trained **solely on a small subset of observed data points**, without any physical constraints. This model serves as a baseline to demonstrate how traditional machine learning struggles to generalize or extrapolate outside the exact zone where training data is available.

2. **Physics-Informed FCNN (FCNN PINN):** This configuration shares the same underlying architecture as the baseline network (using Tanh activations) but incorporates the physics of the system. By **integrating the ODE residual directly into the loss function**, the network is forced to respect the laws of motion. While it shows a massive improvement over the purely data-driven model, standard FCNNs still suffer from *spectral bias*, meaning they struggle to learn high-frequency periodic functions.

3. **Physics-Informed SIREN (Sinusoidal Representation Network):** To overcome the limitations and spectral bias of traditional networks, we implemented a SIREN architecture within the PINN framework. Instead of standard activations, SIREN utilizes **periodic sine functions as activation functions**. This allows the network to naturally capture complex, high-frequency oscillations.

## Training Strategy
The models are trained using a composite loss function that combines:
- **Data Loss:** Mean Squared Error (MSE) comparing the network's prediction with a small set of exact analytical data points during the initial training phase.
- **Physics Loss:** The residual of the ODE $\left(\ddot{\theta}_{pred} + \omega^2\sin\theta_{pred}\right)$ evaluated at a dense set of physical collocation points across the entire time domain.

## Acknowledgments
- Credits to **Vincent Sitzmann** (`vsitzmann` on GitHub) for the formulation and original implementation ideas of the SIREN neural network architecture.
- Credits to **Ben Moseley** for the original implementation of the standard FCNN.