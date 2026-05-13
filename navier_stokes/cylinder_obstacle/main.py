"""
Main module to run the Navier-Stokes PINN simulation with a cylinder.
"""

from solver import CylinderPINN
from utils import plot_flow_fields_cylinder

def main():
    # 1. Configuration Parameters
    R = 0.05
    nu = 0.01
    u_max = 1.0
    h_max = 0.2
    
    # 2. Initialize and Train the Solver
    print("=== Initializing Cylinder PINN Simulator ===")
    pinn_solver = CylinderPINN(R=R, nu=nu, u_max=u_max, h_max=h_max)
    
    pinn_solver.build_and_train(
        iter_adam1=10000, 
        iter_adam2=15000, 
        save_path="pinn_cylinder_model"
    )

    # 3. Generate Evaluation Points and Predict
    print("\n=== Generating Flow Fields Plot ===")
    samples = pinn_solver.geom.random_points(500000)
    result = pinn_solver.predict(samples)

    # 4. Plot results using our utility function
    plot_flow_fields_cylinder(samples, result, R=R)

if __name__ == "__main__":
    main()