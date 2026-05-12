# Navier-Stokes fluid with a cylindrical obstacle

## Problem overview

Simulation of a fluid following the 2D incompressible, laminar, steady-state Navier Stokes equations (NSEs). The equations are, respectively, mass conservation, $x$ and $y$ direction momentum conservation.

$$
\begin{gather*}
    \dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0\\
    u\dfrac{\partial u}{\partial x} + v\dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial x}+ \nu \left(\dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} \right)\\
    u\dfrac{\partial v}{\partial x} + v\dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho}\dfrac{\partial p}{\partial y}+ \nu \left(\dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} \right)
\end{gather*}
$$

Or written in a concise way,

$$
\begin{gather*}
\nabla \cdot \boldsymbol{u} = 0\\
(\boldsymbol{u}\cdot \nabla)\boldsymbol{u}=-\nabla p+\nu \nabla^2\boldsymbol{u}
\end{gather*}
$$

Where $u$ is the velocity in the $x$ direction, $v$ in the $y$ direction, $p$ is the pressure, $\rho$ the density and $\nu$ the kinematic viscosity. 

The system consists of a main 2D rectangular channel of length $L=1~m$ (from $x=-0.2$ to $0.8$) and width $d = 0.4~m$. Inside the channel, there is a cylindrical obstacle of radius $R=0.05~m$ located exactly at the origin $(0,0)$. The fluid has density $\rho = 1~kg/m^3$ and we have studied the case $\nu=0.01~m^2/s$. 

The boundary conditions are the following: 
- At the entrance (inlet), $u=u_{\text{inlet}}$ and $v=0$.
- At the exit (outlet), $p=0$.
- At the walls (including the cylinder boundary), $u=v=0$.

The function $u=u_{\text{inlet}}$ is a parabola that satisfies $u=0$ at the top and bottom walls of the entrance. This is done because, if $u$ was uniform at every point of the inlet, there would be a conflict with the no-slip condition ($u=0$ at the walls). This discontinuity would cause the gradients of the neural network to explode. The inlet function is defined as the following parabola:

$$
\begin{equation*}
u_{\text{inlet}} = u_{0}\left(1 - \left(\dfrac{y}{h_{max}}\right)^2\right)
\end{equation*}
$$

where $u_0$ is the maximum velocity (set to 1 $m/s$ in this case) at the center of the entrance and $h_{max}$ is half the height of the main channel ($0.2~m$).

## DeepXDE Implementation

We define the partial differential equations at the `pde(X,Y)` function, and use the DeepXDE library to build the Physics-Informed Neural Network (PINN). The geometry is defined using Constructive Solid Geometry (CSG) by subtracting the disk (cylinder) from the main rectangular domain.

To accurately capture the complex flow dynamics and the wake generated behind the cylinder, we employ several advanced training strategies:

1. **Point Distribution & Anchors:** We sample $4000$ points inside the domain and $1500$ at the boundaries. To force the network to strictly respect the obstacle and accurately model the boundary layer, we add $3000$ extra anchor points. These anchors are dynamically sampled within an annular region around the cylinder, with a radius ranging from $R$ to $3R$.
2. **Custom Loss Weights:** We assign specific weights to different components of the loss function. The boundary conditions of the cylinder and the channel walls are penalized more heavily ($w=20.0$ and $w=5.0$, respectively) compared to the PDE residual ($w=1.0$). This ensures strict adherence to the no-slip conditions on the solid surfaces.
3. **Multi-Stage Training:** The network is trained in three phases to ensure convergence:
   - Adam optimizer with a learning rate of $10^{-3}$ for $10,000$ iterations.
   - Adam optimizer with a reduced learning rate of $5 \times 10^{-4}$ for $15,000$ iterations.
   - L-BFGS optimizer for final high-precision fine-tuning.