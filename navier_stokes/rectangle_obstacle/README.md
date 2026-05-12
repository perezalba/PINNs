# Navier-Stokes fluid with a rectangular obstacle

## Case Overview
This simulation introduces a rectangular obstacle to evaluate the PINN's performance in a domain with **sharp corners**. Unlike the cylindrical case, the right angles of the rectangle create critical points for the fluid physics, making the learning process for the neural network more challenging.

Inside the main channel, a rectangular obstacle of $0.4 \times 0.1~m$ is placed starting at $x=0$.

**Boundary Conditions:**
- At the entrance ($x=-0.2$): Parabolic $u=u_{\text{inlet}}$ and $v=0$.
- At the exit ($x=0.8$): $p=0$.
- At the channel walls **and all four sides of the rectangle**: No-slip condition ($u=v=0$).

## DeepXDE Implementation Details
The geometry is defined using Constructive Solid Geometry (CSG) by subtracting the small rectangle from the main channel.

To ensure the network correctly captures the flow around the sharp edges of the obstacle, we use the following setup:

1. **Point Distribution & Anchors:** We sample **$4000$ points** in the domain and **$1500$** at the boundaries. To resolve the flow at the corners, we add **$3000$ extra anchor points** specifically distributed in buffer zones (margins of $0.05~m$) around the four faces of the rectangle.
2. **Custom Loss Weights:** We apply high penalties to the boundary conditions to ensure the no-slip condition is met despite the sharp geometry. The obstacle boundaries carry a weight of **$w=20.0$**, while the channel walls have a weight of **$w=5.0$**.
3. **Multi-Stage Training:**
   - Adam optimizer ($lr=10^{-3}$) for $10,000$ iterations.
   - Adam optimizer ($lr=5 \times 10^{-4}$) for $15,000$ iterations.
   - L-BFGS optimizer for final refinement.