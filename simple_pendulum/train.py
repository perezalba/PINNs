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
        theta0 (float): Initial angle (amplitude) in radians.
        n_osc (int): Number of complete oscillations to fit in t_max.
        trn_pts (int): Number of points used for training.
        trn_part (int): Maximum index for training points.
        t_max (float): Total time of the simulation window. Defaults to 0.8.
        epochs (int, optional): Number of training epochs. Defaults to 20000.
    """
    def __init__(self, theta0, n_osc, trn_pts, trn_part, t_max=0.8, epochs=20000):
        self.theta0 = theta0
        self.n_osc = n_osc
        self.t_max = t_max
        self.trn_pts = trn_pts
        self.trn_part = trn_part
        self.epochs = epochs
        
        self.t_per = self.t_max / self.n_osc
        self.w = 2 * np.pi / self.t_per
        self.L = 9.81 / (self.w ** 2)

        self.t = torch.linspace(0, self.t_max, 500).view(-1, 1)
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
        
    def get_plot_data(self):
        """
        Evaluates the trained model on the full time range and returns a dictionary with 
        the evaluation results.
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self, 'use_siren') and self.use_siren:
                theta_pred, _ = self.model(self.t)
            else:
                theta_pred = self.model(self.t)
                
        return {
            't': self.t.numpy(),
            'theta_exact': self.theta.numpy(),
            't_trn': self.t_trn.numpy(),
            'theta_trn': self.theta_trn.numpy(),
            'theta_pred': theta_pred.numpy(),
            'n_osc': self.n_osc,
            'L': self.L
        }

class PendulumPINN():
    """
    Class to configure and train a Physics-Informed Neural Network (PINN).
    
    Args:
        theta0 (float): Initial angle (amplitude) in radians.
        n_osc (int): Number of complete oscillations to fit in t_max.
        trn_pts (int): Number of points used for training.
        trn_part (int): Maximum index for training points.
        t_phys_pts (int): Number of collocation points.
        t_max (float): Total time of the simulation window. Defaults to 0.8.
        epochs (int, optional): Number of training epochs. Defaults to 6000.
        use_siren (bool, optional): If True, uses SIREN. Defaults to False.
    """
    def __init__(self, theta0, n_osc, trn_pts, trn_part, t_phys_pts, t_max=0.8, epochs=6000, use_siren=False):
        self.theta0 = theta0
        self.n_osc = n_osc
        self.t_max = t_max
        self.trn_pts = trn_pts
        self.trn_part = trn_part
        self.t_phys_pts = t_phys_pts
        self.epochs = epochs
        self.use_siren = use_siren
        
        self.t_per = self.t_max / self.n_osc
        self.w = 2 * np.pi / self.t_per
        self.L = 9.81 / (self.w ** 2)
        
        self.t = torch.linspace(0, self.t_max, 500).view(-1, 1)
        self.theta = get_analytical_pendulum(self.t, self.w, self.theta0)
        
        indices = np.linspace(0, self.trn_part, self.trn_pts, dtype=int) 
        self.t_trn = self.t[indices]
        self.theta_trn = self.theta[indices]
        
        self.t_physics = torch.linspace(0, self.t_max, self.t_phys_pts).unsqueeze(1)
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
        Executes the PINN training loop computing both the data loss and physics loss.
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

    def get_plot_data(self):
        """
        Evaluates the trained model on the full time range and returns a dictionary with 
        the evaluation results.
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self, 'use_siren') and self.use_siren:
                theta_pred, _ = self.model(self.t)
            else:
                theta_pred = self.model(self.t)
                
        return {
            't': self.t.numpy(),
            'theta_exact': self.theta.numpy(),
            't_trn': self.t_trn.numpy(),
            'theta_trn': self.theta_trn.numpy(),
            'theta_pred': theta_pred.numpy(),
            'n_osc': self.n_osc,
            'L': self.L
        }