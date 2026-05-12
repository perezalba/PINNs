"""
Module containing the definitions of neural network architectures.
Includes a classic FCNN and a SIREN (Sinusoidal Representation Networks) implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class FCN(nn.Module):
    """
    Fully Connected Neural Network.
    Uses the Hyperbolic Tangent (Tanh) activation function in the hidden layers.
    Credits to Ben Moseley for the implementation.

    Args:
        N_INPUT (int): Number of input dimensions.
        N_OUTPUT (int): Number of output dimensions.
        N_HIDDEN (int): Number of neurons in each hidden layer.
        N_LAYERS (int): Total number of layers (including hidden and output layers).
    """
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        """
        Calculates the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Prediction of the network.
        """
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class SineLayer(nn.Module):
    """
    Individual layer for the SIREN network that uses the sine function
    as the activation function. Credits to Vincent Sitzmann for the implementation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, adds a bias to the layer. Default is True.
        is_first (bool, optional): Indicates if it's the first layer of the network (affects initialization). Default is False.
        omega_0 (float, optional): Base frequency for the sine function. Default is 30.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        """
        Initializes the weights of the layer according to the distribution proposed
        in the original SIREN paper, depending on whether it is the first layer or not.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        """
        Calculates the output of the layer with the sinusoidal activation.

        Args:
            input (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Output of the layer with sinusoidal activation.
        """
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        """
        Returns the output of the layer and the intermediate value before applying the sine.

        Args:
            input (torch.Tensor): Input tensor to the layer.

        Returns:
            tuple: (output with sine, intermediate linear value).
        """
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
class Siren(nn.Module):
    """
    SIREN architecture (Sinusoidal Representation Network).
    A fully connected neural network that uses periodic sinusoidal activation functions.

    Args:
        in_features (int): Dimensions of the input.
        hidden_features (int): Number of neurons per hidden layer.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Dimensions of the output.
        outermost_linear (bool, optional): If True, the last layer does not have sinusoidal activation. Default is False.
        first_omega_0 (float, optional): Frequency omega for the first layer. Default is 30.
        hidden_omega_0 (float, optional): Frequency omega for the hidden layers. Default is 30.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        """
        Calculates the prediction of the network and allows retaining gradients over the coordinates.

        Args:
            coords (torch.Tensor): Input tensor (e.g., spatial or temporal coordinates).

        Returns:
            tuple: (Prediction of the network, coordinates with required gradient).
        """
        coords = coords.clone().detach().requires_grad_(True) 
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        """
        Executes a forward pass while retaining the activations of all layers (useful for visualizations).

        Args:
            coords (torch.Tensor): Input tensor.
            retain_grad (bool, optional): If True, retains the gradients of the intermediate activations.

        Returns:
            OrderedDict: Dictionary with the activations of each layer.
        """
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1
        return activations