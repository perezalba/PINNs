"""
Module containing the Physics-Informed Neural Network (PINN) definition
for solving the 2D Navier-Stokes equations around a rectangular obstacle.
"""

import tensorflow as tf
import deepxde as dde
import numpy as np

dde.config.set_default_float("float32")

class RectanglePINN:
    """
    PINN Solver for 2D newtonian incompressible fluid flow around a rectangular obstacle.

    Args:
        nu (float): Kinematic viscosity of the fluid. Defaults to 0.01.
        u_max (float): Maximum inlet velocity. Defaults to 1.0.
        h_max (float): Half-height of the main channel. Defaults to 0.2.
        obs_bounds (tuple): Boundaries of the rectangle (xmin, xmax, ymin, ymax).
                            Defaults to (0.0, 0.4, -0.05, 0.05).
    """
    def __init__(self, nu=0.01, u_max=1.0, h_max=0.2, obs_bounds=(0.0, 0.4, -0.05, 0.05)):
        self.nu = nu
        self.u_max = u_max
        self.h_max = h_max
        self.obs_xmin, self.obs_xmax, self.obs_ymin, self.obs_ymax = obs_bounds
        
        self.geom = None
        self.model = None

    def _u_inlet(self, X):
        y = X[:, 1:2]
        return self.u_max * (1 - (y / self.h_max) ** 2)

    def _pde(self, X, Y):
        """
        Defines the Navier-Stokes equations for incompressible flow in 2D.
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

    def build_and_train(self, iter_adam1=10000, iter_adam2=15000, 
                        save_path="pinn_rectangle_model", load_weights_path=None):
        """
        Sets up the geometry, boundary conditions, and anchors, then trains the model.
        If load_weights_path is provided, it skips training and loads the specified weights.

        Args:
            iter_adam1 (int): Iterations for the first Adam phase. Defaults to 10000.
            iter_adam2 (int): Iterations for the second Adam phase. Defaults to 15000.
            save_path (str): Prefix to save the model.
            load_weights_path (str, optional): Path to a .h5 file to load pre-trained weights.
        """
        # Define geometry
        geom_rect = dde.geometry.Rectangle(xmin=[-0.2, -self.h_max], xmax=[0.8, self.h_max])
        geom_obs = dde.geometry.Rectangle(
            xmin=[self.obs_xmin, self.obs_ymin], 
            xmax=[self.obs_xmax, self.obs_ymax]
        )
        self.geom = dde.geometry.CSGDifference(geom_rect, geom_obs)

        # Generate anchors dynamically around the obstacle
        buffer = 0.05
        x_left = np.random.uniform(self.obs_xmin - buffer, self.obs_xmin, 500)
        y_left = np.random.uniform(self.obs_ymin - buffer, self.obs_ymax + buffer, 500)
        
        x_right = np.random.uniform(self.obs_xmax, self.obs_xmax + buffer, 500)
        y_right = np.random.uniform(self.obs_ymin - buffer, self.obs_ymax + buffer, 500)
        
        x_top = np.random.uniform(self.obs_xmin, self.obs_xmax, 1000)
        y_top = np.random.uniform(self.obs_ymax, self.obs_ymax + buffer, 1000)
        
        x_bottom = np.random.uniform(self.obs_xmin, self.obs_xmax, 1000)
        y_bottom = np.random.uniform(self.obs_ymin - buffer, self.obs_ymin, 1000)

        rectangle_points = np.vstack((
            np.vstack((x_left, y_left)).T,
            np.vstack((x_right, y_right)).T,
            np.vstack((x_top, y_top)).T,
            np.vstack((x_bottom, y_bottom)).T
        ))

        # Boundary condition identifier functions
        def inlet(X, on_boundary): return on_boundary and np.isclose(X[0], -0.2)
        def outlet(X, on_boundary): return on_boundary and np.isclose(X[0], 0.8)
        def walls(X, on_boundary): return on_boundary and (np.isclose(X[1], -0.2) or np.isclose(X[1], 0.2))
        
        def rectangle_boundary(X, on_boundary):
            return on_boundary and (
                np.isclose(X[0], self.obs_xmin) or 
                np.isclose(X[0], self.obs_xmax) or 
                np.isclose(X[1], self.obs_ymin) or 
                np.isclose(X[1], self.obs_ymax)
            )

        # Apply boundary conditions
        bc_in_u = dde.icbc.DirichletBC(self.geom, self._u_inlet, inlet, component=0)
        bc_in_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, inlet, component=1)
        bc_out_p = dde.icbc.DirichletBC(self.geom, lambda X: 0, outlet, component=2)
        bc_wall_u = dde.icbc.DirichletBC(self.geom, lambda X: 0, walls, component=0)
        bc_wall_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, walls, component=1)
        bc_rect_u = dde.icbc.DirichletBC(self.geom, lambda X: 0, rectangle_boundary, component=0)
        bc_rect_v = dde.icbc.DirichletBC(self.geom, lambda X: 0, rectangle_boundary, component=1)

        # Build data and model
        data = dde.data.PDE(
            self.geom, self._pde,
            [bc_in_u, bc_in_v, bc_out_p, bc_wall_u, bc_wall_v, bc_rect_u, bc_rect_v],
            num_domain=4000, num_boundary=1500, anchors=rectangle_points
        )
        
        net = dde.nn.FNN([2] + [64]*5 + [3], "tanh", "Glorot uniform")
        self.model = dde.Model(data, net)

        weights = [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 20.0, 20.0]
        self.model.compile("adam", lr=1e-3, loss_weights=weights)

        # Logic: Load weights vs Train from scratch
        if load_weights_path:
            print(f"Loading pre-trained weights from {load_weights_path}...")
            # We need to initialize the network with a dummy predict before loading weights in DDE
            self.model.predict(np.zeros((1, 2)))
            self.model.net.load_weights(load_weights_path)
            print("Weights loaded successfully. Skipping training.")
        else:
            print(f"Training Phase 1 (Adam lr=1e-3, {iter_adam1} iter)...")
            self.model.train(iterations=iter_adam1)
            
            print(f"Training Phase 2 (Adam lr=5e-4, {iter_adam2} iter)...")
            self.model.compile("adam", lr=5e-4, loss_weights=weights)
            self.model.train(iterations=iter_adam2)
            
            self.model.compile("L-BFGS")
            print("Training Phase 3 (L-BFGS)...")
            self.model.train()
            
            if save_path:
                self.model.save(save_path)
                print(f"Model saved with prefix: {save_path}")

    def predict(self, X):
        return self.model.predict(X)