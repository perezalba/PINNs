# Navier-Stokes fluid without an obstacle (Poiseuille Flow)

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

The system consists of a 2D rectangular channel of length $L=1~m$ and width $d = 0.4~m$. The fluid has density $\rho = 1~kg/m^3$ and we have studied the case $\nu=0.01~m^2/s$. 

The boundary conditions are the following: 
- At the entrance (inlet), $u=u_{\text{inlet}}$ and $v=0$.
- At the exit (outlet), $p=0$.
- At the walls (top and bottom), $u=v=0$.

The function $u=u_{\text{inlet}}$ is a parabola that satisfies that $u=0$ at the top and bottom walls in the entrance. This is done because, if $u$ was the same at every point of the inlet, then there would be a conflict with the third condition ($u=0$ at the wall). This discontinuity would cause the gradient of the neural network to explode. The inlet function is defined as the following parabola:

$$
\begin{equation*}
u_{\text{inlet}} = u_{0}\left(1 - \left(\dfrac{y}{h_{max}}\right)^2\right)
\end{equation*}
$$

where $u_0$ is the maximum velocity (set to 1 $m/s$ in this case) at the center of the entrance and $h_{max}$ is half the height of the rectangular domain ($0.2~m$). 

## DeepXDE Implementation

We define the equations that must be satisfied at the `pde(X,Y)` function, and we use the library DeepXDE for creating the Physics-Informed Neural Network (PINN). We define the geometry of the system as a simple rectangle, and apply the boundary conditions specified previously. 

For the training process, we choose a total of $3000$ points inside the domain, and another $500$ points specifically at the boundaries. Because there are no complex geometries or obstacles, no extra anchor points are needed. 

With these points, we start training the neural network with the Adam optimizer (to quickly descend the loss landscape) and then refine the solution using the L-BFGS optimizer to achieve high precision.