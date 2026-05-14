"""
Main module to run the Navier-Stokes PINN simulation with a cylinder.
"""

import argparse
import numpy as np
from solver import CylinderPINN
from utils import plot_flow_fields_cylinder

def main():
    parser = argparse.ArgumentParser(description="PINN Simulator for Navier-Stokes Flow around a Cylinder")
    parser.add_argument("--load", type=str, default=None, 
                        help="Route to a pre-trained model file (e.g., 'pinn_cylinder_model_weights.h5'). If not provided, the model will be trained from scratch.")
    args = parser.parse_args()

    R = 0.05
    nu = 0.01
    u_max = 1.0
    h_max = 0.2
    
    print("Initializing Cylinder PINN Simulator...")
    pinn_solver = CylinderPINN(R=R, nu=nu, u_max=u_max, h_max=h_max)
    
    if args.load:
        print(f"Loading pre-trained model from: {args.load}")
        
        pinn_solver.model.compile("adam", lr=1e-3)
        pinn_solver.model.predict(np.zeros((1, 2))) 
        
        pinn_solver.model.net.load_weights(args.load)
        print("Weights loaded successfully. Skipping training.")
    
    else:
        print("Starting training from scratch...")
        pinn_solver.build_and_train(
            iter_adam1=10000, 
            iter_adam2=15000, 
            save_path="pinn_cylinder_model"
        )

    print("\nGenerating Flow Fields Plot...")
    samples = pinn_solver.geom.random_points(500000)
    result = pinn_solver.predict(samples)

    plot_flow_fields_cylinder(samples, result, R=R)

if __name__ == "__main__":
    main()