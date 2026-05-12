"""
Main module to run the pendulum experiments.
This script initializes the physical parameters and trains the different
models (standard FCNN and PINNs with/without SIREN), showing the results.

Usage:
    python main.py
"""

import numpy as np
from train import PendulumFCNN, PendulumPINN

def main():
    """
    Main function to define hyperparameters and execute the training and 
    visualization of all models sequentially.
    """
    L = 0.01
    theta0 = 0.6
    n_osc = 4
    trn_pts = 50
    trn_part = 100
    t_phys_pts = 200

    print("=== FCNN estàndard ===")
    model_fcnn = PendulumFCNN(L, theta0, n_osc, trn_pts, trn_part, epochs=2000)
    model_fcnn.train()
    model_fcnn.plot_result()

    print("\n=== PINN (sense SIREN) ===")
    model_pinn_standard = PendulumPINN(L, theta0, n_osc, trn_pts, trn_part, t_phys_pts, epochs=4000, use_siren=False)
    model_pinn_standard.train()
    model_pinn_standard.plot_result()

    print("\n=== PINN (amb SIREN) ===")
    model_pinn_siren = PendulumPINN(L, theta0, n_osc, trn_pts, trn_part, t_phys_pts, epochs=4000, use_siren=True)
    model_pinn_siren.train()
    model_pinn_siren.plot_result()

if __name__ == "__main__":
    main()