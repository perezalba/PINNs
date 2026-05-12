import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj

from networks import FCN, Siren

class PendulumFCNN():
    def __init__(self, L, theta0, n_osc, trn_pts, trn_part, epochs=20000):
        self.L = L
        self.theta0 = theta0
        self.n_osc = n_osc
        self.trn_pts = trn_pts
        self.trn_part = trn_part
        self.epochs = epochs
        
        self.w = np.sqrt(9.81 / self.L)
        self.t_per = 2 * np.pi / self.w

        self.t = torch.linspace(0, self.n_osc * self.t_per, 500).view(-1, 1)
        self.theta = self.analytical_pendulum(self.t)
        
        indices = np.linspace(0, self.trn_part, self.trn_pts, dtype=int) 
        self.t_trn = self.t[indices]
        self.theta_trn = self.theta[indices]
        
        torch.manual_seed(123)
        self.model = FCN(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=32, N_LAYERS=3)

        self.loss_history = []

    def analytical_pendulum(self, t):
        k = np.sin(self.theta0 / 2)
        t_np = t.detach().cpu().numpy()
        u = self.w * (t_np + (np.pi / (2 * self.w)))
        sn, cn, dn, ph = ellipj(u, k**2)
        theta_np = 2 * np.arcsin(k * sn)
        return torch.tensor(theta_np, dtype=torch.float32)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        print(f"Starting standard FCNN training for {self.epochs} epochs.")
        
        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            theta_pred_data = self.model(self.t_trn)
            loss = torch.mean((theta_pred_data - self.theta_trn)**2)
            
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

            if (i + 1) % 1000 == 0:
                print(f"Epoch {i+1:5d} | MSE Loss: {loss.item():.4e}")

        print("Training complete.")
        
    def plot_result(self):
        self.model.eval()
        with torch.no_grad():
            theta_pred_final = self.model(self.t)
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.t.numpy(), self.theta.numpy(), label="Analytical Solution", color='blue', alpha=0.3, lw=2)
        plt.plot(self.t.numpy(), theta_pred_final.numpy(), label="FCNN Prediction", color='blue', linestyle='--')
        plt.scatter(self.t_trn.numpy(), self.theta_trn.numpy(), color='red', s=30, label='Training Data', zorder=5)

        t_max_trn = self.t_trn.max().item()
        plt.axvspan(0, t_max_trn, color='grey', alpha=0.15, label='Training Zone')
        plt.axvspan(t_max_trn, self.t.max().item(), color='orange', alpha=0.05, label='Extrapolation Zone')
        
        plt.title("Standard FCNN Results")
        plt.xlabel("Time (s)")
        plt.ylabel("Theta (rad)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class PendulumPINN():
    def __init__(self, L, theta0, n_osc, trn_pts, trn_part, t_phys_pts, epochs=6000, use_siren=False):
        self.L = L
        self.theta0 = theta0
        self.n_osc = n_osc
        self.trn_pts = trn_pts
        self.trn_part = trn_part
        self.t_phys_pts = t_phys_pts
        self.epochs = epochs
        self.use_siren = use_siren
        
        self.w = np.sqrt(9.81 / self.L)
        self.t_per = 2 * np.pi / self.w
        
        self.t = torch.linspace(0, self.n_osc * self.t_per, 500).view(-1, 1)
        self.theta = self.analytical_pendulum(self.t)
        
        indices = np.linspace(0, self.trn_part, self.trn_pts, dtype=int) 
        self.t_trn = self.t[indices]
        self.theta_trn = self.theta[indices]
        
        self.t_physics = torch.linspace(0, self.n_osc * self.t_per, self.t_phys_pts).unsqueeze(1)
        self.t_physics.requires_grad_(True)
        
        torch.manual_seed(123)
        if self.use_siren:
            self.model = Siren(in_features=1, hidden_features=32, hidden_layers=3, out_features=1, outermost_linear=True, 
                               first_omega_0=self.w, hidden_omega_0=30.)
        else:
            self.model = FCN(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=32, N_LAYERS=3)

        self.loss_history = []

    def analytical_pendulum(self, t):
        k = np.sin(self.theta0 / 2)
        t_np = t.detach().cpu().numpy()
        u = self.w * (t_np + (np.pi / (2 * self.w)))
        sn, cn, dn, ph = ellipj(u, k**2)
        theta_np = 2 * np.arcsin(k * sn)
        return torch.tensor(theta_np, dtype=torch.float32)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000, min_lr=1e-5)

        print(f"Starting training for {self.epochs} epochs. SIREN enabled: {self.use_siren}")
        
        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            if self.use_siren:
                theta_pred_data, _ = self.model(self.t_trn)
                theta_pred_phys, t_grad = self.model(self.t_physics)
            else:
                theta_pred_data = self.model(self.t_trn)
                theta_pred_phys = self.model(self.t_physics)
                t_grad = self.t_physics
            
            loss1 = torch.mean((theta_pred_data - self.theta_trn)**2)
            
            dtheta = torch.autograd.grad(theta_pred_phys, t_grad, torch.ones_like(theta_pred_phys), create_graph=True)[0]
            dtheta2 = torch.autograd.grad(dtheta, t_grad, torch.ones_like(dtheta), create_graph=True)[0]
            
            physics_res = dtheta2 + (self.w**2) * torch.sin(theta_pred_phys)
            loss2 = (1e-6) * torch.mean(physics_res**2)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            self.loss_history.append(loss.item())

            if (i + 1) % 1000 == 0:
                print(f"Epoch {i+1:5d} | Total Loss: {loss.item():.4e}")

        print("Training complete.")

    def plot_result(self):
        with torch.no_grad():
            if self.use_siren:
                theta_pred_final, _ = self.model(self.t)
            else:
                theta_pred_final = self.model(self.t)
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.t.numpy(), self.theta.numpy(), label="Analytical Solution", color='blue', alpha=0.3, lw=2)
        plt.plot(self.t.numpy(), theta_pred_final.numpy(), label="PINN Prediction", color='blue', linestyle='--')
        plt.scatter(self.t_trn.numpy(), self.theta_trn.numpy(), color='red', s=30, label='Training Data', zorder=5)

        t_max_trn = self.t_trn.max().item()
        plt.axvspan(0, t_max_trn, color='grey', alpha=0.15, label='Training Zone')
        plt.axvspan(t_max_trn, self.t.max().item(), color='orange', alpha=0.05, label='Extrapolation Zone')
        
        plt.title(f"PINN Results (SIREN: {self.use_siren})")
        plt.xlabel("Time (s)")
        plt.ylabel("Theta (rad)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()