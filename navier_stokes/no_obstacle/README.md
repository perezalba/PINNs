# Navier-Stokes fluid without an obstacle (Poiseuille Flow)

## Case Overview
This simulation represents the baseline case of the fluid dynamics study: the **Poiseuille flow**. It serves to validate the Physics-Informed Neural Network (PINN) against a well-known analytical solution in a simple geometry.

The domain is a clear 2D rectangular channel where the fluid develops a stable velocity profile driven by the parabolic inlet condition.

**Boundary Conditions:**
- At the entrance ($x=-0.2$): Parabolic $u=u_{\text{inlet}}$ and $v=0$.
- At the exit ($x=0.8$): $p=0$.
- At the top and bottom walls: No-slip condition ($u=v=0$).

## DeepXDE Implementation Details
The geometry is defined as a simple rectangular domain. Since the flow is highly regular and lacks complex gradients or wakes, the training requirements are lower than in cases with obstacles.

1. **Point Distribution:** We sample **$3000$ points** inside the domain and **$500$ points** at the boundaries. No extra anchor points are required as there are no internal boundaries or sharp corners to resolve.
2. **Standard Loss Weights:** Given the simplicity of the domain, a balanced weight distribution ($w=1.0$) is sufficient for the network to converge to the solution.
3. **Training Strategy:** The network is trained using a dual-stage approach:
   - Adam optimizer ($lr=10^{-3}$) for initial convergence.
   - L-BFGS optimizer for final high-precision fine-tuning to match the analytical solution.