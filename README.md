# Physics-Informed Neural Networks (PINNs) for Physical Simulations

This repository contains the code for my Bachelor's Thesis (TFG), focused on exploring the capabilities of Physics-Informed Neural Networks (PINNs) to solve different differential equations governed by physical laws.

The project is structured in two main physical problems of increasing complexity:

## 1. The Simple Pendulum (1D Time-Domain)
A fundamental test case to understand how PINNs learn differential equations compared to standard Data-Driven Fully Connected Neural Networks (FCNNs). We explore standard architectures as well as Sinusoidal Representation Networks (SIRENs) to capture periodic motion.
* [Pendulum Simulations](./simple_pendulum/main.py)

## 2. Incompressible Fluid Flow (2D Navier-Stokes)
An advanced application solving the Navier-Stokes equations for fluid dynamics using the **DeepXDE** library. 

### Fluid Mechanics: General Framework
All fluid simulations in this repository follow the 2D incompressible, laminar, steady-state Navier-Stokes Equations (NSEs). These comprise mass conservation and momentum conservation in the $x$ and $y$ directions:

$$
\begin{gather*}
\nabla \cdot \boldsymbol{u} = 0\\
(\boldsymbol{u}\cdot \nabla)\boldsymbol{u}=-\nabla p+\nu \nabla^2\boldsymbol{u}
\end{gather*}
$$

Expanded into spatial coordinates:

$$
\begin{gather*}
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0\\
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x}+ \nu \left(\dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} \right)\\
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y}+ \nu \left(\dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} \right)
\end{gather*}
$$

Where $u$ is the horizontal velocity, $v$ the vertical velocity, $p$ the pressure, $\rho$ the density ($1~kg/m^3$), and $\nu$ the kinematic viscosity ($0.01~m^2/s$). 

**General Domain & Inlet Conditions:**
All fluid experiments are simulated inside a rectangular channel of $1~m$ length (from $x=-0.2$ to $0.8$) and width $d = 0.4~m$. 

To prevent gradient explosion caused by discontinuities at the corners, the inlet velocity ($u_{\text{inlet}}$) is modeled as a parabola ensuring the no-slip condition ($u=0$) at the walls:
$$
\begin{equation*}
u_{\text{inlet}} = u_{0}\left(1 - \left(\dfrac{y}{h_{max}}\right)^2\right)
\end{equation*}
$$
where $u_0 = 1~m/s$ is the maximum velocity at the center and $h_{max} = 0.2~m$ is half the height of the channel.

### Explored Geometries:
Using this general framework, three different scenarios have been modeled:
* [Case A: Poiseuille Flow (No Obstacle)](./navier_stokes/no_obstacle/main.py)
* [Case B: Flow around a Cylinder](./navier_stokes/cylinder_obstacle/main.py)
* [Case C: Flow around a Rectangle](./navier_stokes/rectangle_obstacle/main.py)

---
## Installation & Requirements
To run the code in this repository locally, you will need a Python environment with the following main libraries:
* `torch` (PyTorch) for the Pendulum simulations.
* `deepxde` and `tensorflow` for the Fluid simulations.
* `numpy`, `scipy`, and `matplotlib` for mathematics and plotting.