# Navier-Stokes fluid with a cylindrical obstacle

## Case Overview
This simulation builds upon the general Navier-Stokes framework defined in the main repository, introducing a solid obstacle to evaluate the PINN's ability to capture complex flow dynamics and wake generation.

Inside the main channel, a cylindrical obstacle of radius $R=0.05~m$ is located exactly at the origin $(0,0)$. 

**Boundary Conditions:**
- At the entrance ($x=-0.2$): Parabolic $u=u_{\text{inlet}}$ and $v=0$.
- At the exit ($x=0.8$): $p=0$.
- At the channel walls **and the cylinder surface**: No-slip condition ($u=v=0$).

## DeepXDE Implementation Details
The geometry is defined using Constructive Solid Geometry (CSG) by subtracting the disk (cylinder) from the main rectangular domain.

To accurately model the boundary layer around the cylinder, we employ several advanced training strategies:

1. **Point Distribution & Anchors:** We sample $4000$ points inside the domain and $1500$ at the boundaries. To force the network to strictly respect the solid obstacle, we add **$3000$ extra anchor points**. These anchors are dynamically sampled within an annular region around the cylinder, with a radius ranging from $R$ to $3R$.
2. **Custom Loss Weights:** We assign specific weights to different components of the loss function. The boundary conditions of the cylinder and the channel walls are penalized more heavily ($w=20.0$ and $w=5.0$, respectively) compared to the PDE residual ($w=1.0$).
3. **Multi-Stage Training:** The network is trained in three phases:
   - Adam optimizer ($lr=10^{-3}$) for $10,000$ iterations.
   - Adam optimizer ($lr=5 \times 10^{-4}$) for $15,000$ iterations.
   - L-BFGS optimizer for final convergence.