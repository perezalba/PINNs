"""
Main entry point for running the Navier-Stokes PINN simulation with a rectangular obstacle.
"""

from navier_stokes.rectangle_obstacle.solver import RectanglePINN
from navier_stokes.rectangle_obstacle.utils import plot_flow_fields_rectangle

def main():
    # 1. Configuration Parameters
    nu = 0.01
    u_max = 1.0
    h_max = 0.2
    
    # Rectangle boundaries: (xmin, xmax, ymin, ymax)
    obs_bounds = (0.0, 0.4, -0.05, 0.05)
    
    # 2. Initialize the Solver
    print("=== Rectangle PINN Simulator ===")
    pinn_solver = RectanglePINN(nu=nu, u_max=u_max, h_max=h_max, obs_bounds=obs_bounds)
    
    # 3. Train or Load Model
    pinn_solver.build_and_train(
        iter_adam1=10000, 
        iter_adam2=15000, 
        save_path="pinn_rectangle_model",
        load_weights_path=None 
    )

    # 4. Generate Evaluation Points and Predict
    print("\n=== Generating Flow Fields Plot ===")
    samples = pinn_solver.geom.random_points(500000)
    result = pinn_solver.predict(samples)

    xmin, xmax, ymin, ymax = obs_bounds
    width = xmax - xmin
    height = ymax - ymin
    matplotlib_coords = (xmin, ymin, width, height)

    # 5. Plot results
    plot_flow_fields_rectangle(samples, result, obs_coords=matplotlib_coords)

if __name__ == "__main__":
    main()