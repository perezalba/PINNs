"""
Module responsible for handling the training loops of the pendulum models.
Includes classes to train purely data-driven models (FCNN) and physics-informed models (PINN).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj

from networks import FCN, Siren

from utils import get_analytical_pendulum

class PendulumFCNN():
    """
    Class defined for configuring and training a Fully-Connected Neural
    Network (FCNN) for predicting the dynamics of a simple pendulum.

    Args:
        L (float): Length of the pendulum in meters.
        theta0 (float): Initial angle (amplitude) in radians.
        n_osc (int): Number of complete oscillations to simulate.
        trn_pts (int): Number of points used for training.
        trn_part (int): Maximum index for training points.
        epochs (int, optional): Number of training epochs. Defaults to 20000.
    """
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
        self.theta = get_analytical_pendulum(self.t, self.w, self.theta0)
        
        indices = np.linspace(0, self.trn_part, self.trn_pts, dtype=int) 
        self.t_trn = self.t[indices]
        self.theta_trn = self.theta[indices]
        
        torch.manual_seed(123)
        self.model = FCN(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=32, N_LAYERS=3)

        self.loss_history = []

    def train(self):
        """
        Trains the FCNN model using Mean Squared Error (MSE) loss on the training data.
        """
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
        """
        Plots the analytical solution, the network's prediction, and the training 
        data points, highlighting the training and extrapolation zones.
        """
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
    """
    Class to configure and train a Physics-Informed Neural Network (PINN) for a simple pendulum.
    Allows choosing between a standard FCNN architecture or a SIREN architecture.

    Args:
        L (float): Length of the pendulum in meters.
        theta0 (float): Initial angle (amplitude) in radians.
        n_osc (int): Number of complete oscillations to simulate.
        trn_pts (int): Number of points used for training.
        trn_part (int): Maximum index for training points.
        t_phys_pts (int): Number of collocation points used to evaluate the physics residual.
        epochs (int, optional): Number of training epochs. Defaults to 6000.
        use_siren (bool, optional): If True, uses the SIREN architecture instead of standard FCNN. Defaults to False.
    """
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
        self.theta = get_analytical_pendulum(self.t, self.w, self.theta0)
        
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

    def train(self):
        """
        Executes the PINN training loop computing both the data loss (MSE on observed points) 
        and the physics loss (residual of the pendulum differential equation on collocation points).
        Calculates and prints the final RMSE over the entire trajectory.
        """
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
        """
        Plots the analytical solution, the network's prediction, and the training data points,
        highlighting the training and extrapolation zones.
        """
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