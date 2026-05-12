import tensorflow as tf
import deepxde as dde
import numpy as np

class FluidPINN:
    def __init__(self, rho=1.0, mu=0.01, u_max=1.0, h_max=0.2):
        self.rho = rho
        self.mu = mu
        self.u_max = u_max
        self.h_max = h_max
        
        self.geom = None
        self.model = None

    def _u_inlet(self, x):
        y = x[:, 1:2]
        return self.u_max * (1 - (y / self.h_max) ** 2)

    def _pde(self, X, Y):
        du_x = dde.grad.jacobian(Y, X, i=0, j=0)
        du_y = dde.grad.jacobian(Y, X, i=0, j=1)
        dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
        dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
        dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
        dp_y = dde.grad.jacobian(Y, X, i=2, j=1)

        du_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)
        dv_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)
        dv_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)

        pde_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + 1 / self.rho * dp_x - (self.mu / self.rho) * (du_xx + du_yy)
        pde_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + 1 / self.rho * dp_y - (self.mu / self.rho) * (dv_xx + dv_yy)
        pde_cont = du_x + dv_y

        return [pde_u, pde_v, pde_cont]

    def build_and_train(self, iterations=10000, save_path="pinn_no_obstacle_model"):
        self.geom = dde.geometry.Rectangle(xmin=[-0.2, -self.h_max], xmax=[0.8, self.h_max])

        def boundary_wall(X, on_boundary):
            return on_boundary and np.logical_or(
                np.isclose(X[1], -self.h_max, rtol=1e-05, atol=1e-08),
                np.isclose(X[1], self.h_max, rtol=1e-05, atol=1e-08)
            )

        def boundary_inlet(X, on_boundary):
            return on_boundary and np.isclose(X[0], -0.2, rtol=1e-05, atol=1e-08)

        def boundary_outlet(X, on_boundary):
            return on_boundary and np.isclose(X[0], 0.8, rtol=1e-05, atol=1e-08)

        bc_wall_u = dde.icbc.DirichletBC(self.geom, lambda X: 0.0, boundary_wall, component=0)
        bc_wall_v = dde.icbc.DirichletBC(self.geom, lambda X: 0.0, boundary_wall, component=1)
        bc_inlet_u = dde.icbc.DirichletBC(self.geom, self._u_inlet, boundary_inlet, component=0)
        bc_inlet_v = dde.icbc.DirichletBC(self.geom, lambda X: 0.0, boundary_inlet, component=1)
        bc_outlet_p = dde.icbc.DirichletBC(self.geom, lambda X: 0.0, boundary_outlet, component=2)

        data = dde.data.PDE(
            self.geom,
            self._pde,
            [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p],
            num_domain=3000,
            num_boundary=500
        )

        net = dde.nn.FNN([2] + [64] * 5 + [3], "tanh", "Glorot uniform")
        self.model = dde.Model(data, net)

        self.model.compile("adam", lr=1e-3)
        print("Starting Adam optimization...")
        self.model.train(iterations=iterations)
        
        self.model.compile("L-BFGS")
        print("Starting L-BFGS optimization...")
        self.model.train()

        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call build_and_train() first.")
        return self.model.predict(X)