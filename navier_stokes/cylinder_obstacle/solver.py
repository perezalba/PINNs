"""
Module containing the Physics-Informed Neural Network (PINN) definition
for solving the 2D Navier-Stokes equations around a cylindrical obstacle.
"""

import tensorflow as tf
import deepxde as dde
import numpy as np

dde.config.set_default_float("float32")

class CylinderPINN:
    """
    PINN Solver for 2D newtonian incompressible fluid flow around a cylinder.

    Args:
        R (float): Radius of the cylinder. Defaults to 0.05.
        nu (float): Kinematic viscosity of the fluid. Defaults to 0.01.
        u_max (float): Maximum inlet velocity. Defaults to 1.0.
        h_max (float): Half-height of the channel. Defaults to 0.2.
    """
    def __init__(self, R=0.05, nu=0.01, u_max=1.0, h_max=0.2):
        self.R = R
        self.nu = nu
        self.u_max = u_max
        self.h_max = h_max
        
        self.geom = None
        self.model = None

    def _u_inlet(self, X):
        y = X[:, 1:2]
        return self.u_max * (1 - (y / self.h_max) ** 2)

    def _pde(self, X, Y):
        """
        Defines the Navier-Stokes equations for incompressible flow in 2D
        using kinematic viscosity.

        Args:
            X: Input coordinates [x, y].
            Y: Network output [u, v, p].

        Returns:
            list: Residuals of [continuity, x-momentum, y-momentum].
        """
        u = Y[:, 0:1]
        v = Y[:, 1:2]

        u_x = dde.grad.jacobian(Y, X, i=0, j=0)
        u_y = dde.grad.jacobian(Y, X, i=0, j=1)
        v_x = dde.grad.jacobian(Y, X, i=1, j=0)
        v_y = dde.grad.jacobian(Y, X, i=1, j=1)
        p_x = dde.grad.jacobian(Y, X, i=2, j=0)
        p_y = dde.grad.jacobian(Y, X, i=2, j=1)

        u_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        return [continuity, momentum_x, momentum_y]

    def build_and_train(self, iter_adam1=10000, iter_adam2=15000, save_path="pinn_cylinder_model"):
        """
        Sets up the geometry, obstacle, anchors, boundary conditions, and trains the model 
        using a multi-stage approach (Adam high LR -> Adam low LR -> L-BFGS).

        Args:
            iter_adam1 (int): Iterations for the first Adam optimizer phase. Defaults to 10000.
            iter_adam2 (int): Iterations for the second Adam optimizer phase. Defaults to 15000.
            save_path (str): File prefix to save the model. Defaults to "pinn_cylinder_model".
        """
        # Define geometry with CSG difference
        geom_rect = dde.geometry.Rectangle(xmin=[-0.2, -0.2], xmax=[0.8, 0.2])
        geom_cyl = dde.geometry.Disk([0.0, 0.0], self.R)
        self.geom = dde.geometry.CSGDifference(geom_rect, geom_cyl)

        # Generate anchor points around the cylinder
        num_points_cylinder = 3000
        r_rand = np.random.uniform(self.R, 3 * self.R, num_points_cylinder)
        theta_rand = np.random.uniform(0, 2 * np.pi, num_points_cylinder)
        x_rand = r_rand * np.cos(theta_rand)
        y_rand = r_rand * np.sin(theta_rand)
        cylinder_anchors = np.vstack((x_rand, y_rand)).T

        # Define boundary locations
        def inlet(X, on_boundary):
            return on_boundary and np.isclose(X[0], -0.2)

        def outlet(X, on_boundary):
            return on_boundary and np.isclose(X[0], 0.8)

        def walls(X, on_boundary):
            return on_boundary and (np.isclose(X[1], -0.2) or np.isclose(X[1], 0.2))

        def cylinder(X, on_boundary):
            r = np.sqrt(X[0]**2 + X[1]**2)
            return on_boundary and np.isclose(r, self.R, atol=1e-3)

        # Define Dirichlet Boundary Conditions
        bc_in_u = dde.icbc.DirichletBC(self.geom, self._u_inlet, inlet, component=0)
        bc_in_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, inlet, component=1)
        bc_out_p = dde.icbc.DirichletBC(self.geom, lambda X: 0, outlet, component=2)
        bc_wall_u = dde.icbc.DirichletBC(self.geom, lambda X: 0, walls, component=0)
        bc_wall_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, walls, component=1)
        bc_cyl_u = dde.icbc.DirichletBC(self.geom, lambda X: 0, cylinder, component=0)
        bc_cyl_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, cylinder, component=1)

        # Prepare Data
        data = dde.data.PDE(
            self.geom,
            self._pde,
            [bc_in_u, bc_in_v, bc_out_p, bc_wall_u, bc_wall_v, bc_cyl_u, bc_cyl_v],
            num_domain=4000,
            num_boundary=1500,
            anchors=cylinder_anchors
        )

        # Network architecture
        net = dde.nn.FNN([2] + [64]*5 + [3], "tanh", "Glorot uniform")
        self.model = dde.Model(data, net)

        # Custom loss weights (3 PDEs + 7 BCs = 10 weights)
        weights = [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 20.0, 20.0]

        # Training Phase 1: Adam high LR
        print(f"Starting Adam optimization (Phase 1, lr=1e-3, {iter_adam1} iter)...")
        self.model.compile("adam", lr=1e-3, loss_weights=weights)
        self.model.train(iterations=iter_adam1)

        # Training Phase 2: Adam low LR
        print(f"Starting Adam optimization (Phase 2, lr=5e-4, {iter_adam2} iter)...")
        self.model.compile("adam", lr=5e-4, loss_weights=weights)
        self.model.train(iterations=iter_adam2)

        # Training Phase 3: L-BFGS
        print("Starting L-BFGS optimization (Phase 3)...")
        self.model.compile("L-BFGS")
        self.model.train()

        # Save model
        if save_path:
            self.model.save(save_path)
            print(f"Model saved with prefix: {save_path}")

    def predict(self, X):
        """
        Evaluates the trained network.

        Args:
            X (numpy.ndarray): Coordinate points.

        Returns:
            numpy.ndarray: Predictions [u, v, p].
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)