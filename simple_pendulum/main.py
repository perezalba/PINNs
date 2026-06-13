"""
Main module to run the pendulum experiments.
This script initializes the physical parameters and trains the different
models (standard FCNN and PINNs with/without SIREN), showing the results.
"""

import numpy as np
from train import PendulumFCNN, PendulumPINN
from utils import plot_results

def main():
    """
    Main function to define hyperparameters and execute the training and 
    visualization of all models sequentially.
    """
    theta0 = 0.6
    trn_pts = 50
    trn_part = 100
    t_phys_pts = 200

    print("\nStandard FCNN")
    results_fcnn = []
    n_osc_list = [4,6,8,10,12]
    for n in n_osc_list:
        print(f"\n-> Training FCNN for {n} oscillations...")
        model = PendulumFCNN(theta0, n_osc=n, trn_pts=trn_pts, trn_part=trn_part, epochs=10000)
        model.train()
        results_fcnn.append(model.get_plot_data())
        
    plot_results(results_fcnn, "Standard FCNN", "fcnn_standard_results.png")

    print("\nPINN (without SIREN)")
    results_pinn_std = []
    n_osc_list_pinn_std = [4,5,6,7,8]
    for n in n_osc_list_pinn_std:
        print(f"\n-> Training PINN (without SIREN) for {n} oscillations...")
        model = PendulumPINN(theta0, n_osc=n, trn_pts=trn_pts, trn_part=trn_part, t_phys_pts=t_phys_pts, epochs=30000, use_siren=False)
        model.train()
        results_pinn_std.append(model.get_plot_data())
        
    plot_results(results_pinn_std, "Standard PINN", "pinn_standard_results.png")

    print("\nPINN (with SIREN)")
    results_pinn_siren = []
    n_osc_list_siren = [4,6,8,10,12]
    for n in n_osc_list_siren:
        print(f"\n-> Training PINN (SIREN) for {n} oscillations...")
        model = PendulumPINN(theta0, n_osc=n, trn_pts=trn_pts, trn_part=trn_part, t_phys_pts=t_phys_pts, epochs=3000, use_siren=True)
        model.train()
        results_pinn_siren.append(model.get_plot_data())
        
    plot_results(results_pinn_siren, "SIREN PINN", "pinn_siren_results.png")

if __name__ == "__main__":
    main()