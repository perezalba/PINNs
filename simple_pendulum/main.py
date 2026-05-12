import numpy as np
from train import PendulumFCNN, PendulumPINN

def main():
    L = 1.0
    theta0 = np.pi / 4
    n_osc = 4
    trn_pts = 40
    trn_part = 49
    t_phys_pts = 1000

    print("=== FCNN estàndard ===")
    model_fcnn = PendulumFCNN(L, theta0, n_osc, trn_pts, trn_part, epochs=2000)
    model_fcnn.train()
    model_fcnn.plot_result()

    print("\n=== PINN (sense SIREN) ===")
    model_pinn_standard = PendulumPINN(L, theta0, n_osc, trn_pts, trn_part, t_phys_pts, epochs=2000, use_siren=False)
    model_pinn_standard.train()
    model_pinn_standard.plot_result()

    print("\n=== PINN (amb SIREN) ===")
    model_pinn_siren = PendulumPINN(L, theta0, n_osc, trn_pts, trn_part, t_phys_pts, epochs=2000, use_siren=True)
    model_pinn_siren.train()
    model_pinn_siren.plot_result()

if __name__ == "__main__":
    main()