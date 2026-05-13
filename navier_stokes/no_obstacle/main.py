"""
Main module to run the Navier-Stokes PINN simulation.
"""

import numpy as np
from solver import FluidPINN
from utils import plot_flow_fields, plot_velocity_profile, get_poiseuille_velocity

def main():
    # 1. Configuration & Physical Parameters
    rho = 1.0
    mu = 0.01
    u_max = 1.0
    H = 0.4        # Total height
    h_max = H / 2  
    
    # 2. Initialize and Train Solver
    print("=== Initializing Physics-Informed Neural Network ===")
    pinn_solver = FluidPINN(rho=rho, mu=mu, u_max=u_max, h_max=h_max)
    pinn_solver.build_and_train(iterations=10000, save_path="pinn_no_obstacle_model")

    # 3. Generate Flow Field Plots (Whole domain)
    print("\n=== Generating Flow Fields Plot ===")
    samples = pinn_solver.geom.random_points(500000)
    result = pinn_solver.predict(samples)
    plot_flow_fields(samples, result)

    # 4. Generate Velocity Profile Plot at specific x
    print("\n=== Generating Velocity Profile at x=0.5 ===")
    x_eval = 0.5
    num_points = 100

    y_eval = np.linspace(-h_max, h_max, num_points).reshape(-1, 1)
    x_array = np.full_like(y_eval, x_eval)
    X_test = np.hstack((x_array, y_eval))

    u_pred = pinn_solver.predict(X_test)[:, 0]

    y_teoric = np.linspace(-h_max, h_max, 500)
    u_teoric = get_poiseuille_velocity(y_teoric, u_max, h_max)

    plot_velocity_profile(y_eval, u_pred, y_teoric, u_teoric, x_eval, H)

if __name__ == "__main__":
    main()