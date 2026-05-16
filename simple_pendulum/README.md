# 1D Simple Pendulum

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
- **Physics Loss:** The residual of the ODE $\left(\ddot{\theta}_{\text{pred}} + \omega^2\sin\theta_{\text{pred}}\right)$ evaluated at a dense set of physical collocation points across the entire time domain.

## Experiment Design: Testing Spectral Bias
To explicitly demonstrate the *spectral bias* of standard architectures versus the capabilities of SIREN, the project includes an automated experimental pipeline:
- We fix a **constant time window** ($t_{\text{max}} = 0.8$ seconds) for all models.
- We run sequential training experiments requiring the network to learn $N$ complete oscillations (e.g., $N=2, 4, 8$) within this same time window.
- To achieve this physically, the pipeline dynamically adjusts the pendulum's length ($L$) and angular frequency ($\omega$) for each experiment. 

This setup clearly visualizes how standard PINNs fail to capture high-frequency dynamics as the oscillations become denser, while the SIREN architecture maintains high predictive accuracy.

## Usage & Outputs
To run the full suite of experiments (Standard FCNN, Standard PINN, and SIREN PINN), simply execute the main script:

```bash
python main.py
```

The script will sequentially train a new model for each target frequency. Once the training loops are completed, the pipeline generates comparative plots for each architecture and saves them as high-resolution `.png` files inside the `plots/` directory.

## Acknowledgments
- Credits to **Vincent Sitzmann** (`vsitzmann` on GitHub) for the formulation and original implementation ideas of the SIREN neural network architecture.
- Credits to **Ben Moseley** for the original implementation of the standard FCNN.