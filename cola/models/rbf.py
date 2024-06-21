import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PyTorchRBFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PyTorchRBFNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.name = "rbf"

        # Parameters for the RBF layer
        self.centers = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.sigmas = nn.Parameter(torch.Tensor(hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

        # Initialize parameters
        self.init_params()

    def init_params(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.uniform_(self.sigmas, 0.5, 1.5)  # Adjusted initialization

    def forward(self, x):
        # Calculate RBF layer outputs
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)

        # Numerical stability: add a small constant to prevent division by zero
        out = torch.exp(-dist_sq / (2 * self.sigmas**2 + 1e-6))

        # Linear layer
        out = self.output(out)

        # Check for NaN in the output and replace with a default value (e.g., 0)
        if torch.isnan(out).any():
            # Handling NaN values - can choose to set to a specific value or handle differently
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        return out

    def predict(self, x):
        x = torch.FloatTensor(x)
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()

    def predict_proba(self, x):
        x = torch.FloatTensor(x)
        return self(x).reshape(-1).detach().numpy()

    def predict(self, x):
        # Ensure input is a torch.FloatTensor
        if not isinstance(x, torch.FloatTensor):
            x = torch.FloatTensor(x)

        # Forward pass through the network
        with torch.no_grad():
            output = self(x)

        # Check for NaN in the output and replace with a default value (e.g., 0)
        if torch.isnan(output).any():
            # Handling NaN values - can choose to set to a specific value or handle differently
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

        # Apply your decision threshold
        return (output.reshape(-1) > 0.5).float().numpy()

    def predict_proba(self, x):
        x = torch.FloatTensor(x)
        # Forward pass to get output probabilities for class 1
        probs_class1 = self(x).reshape(-1).detach().numpy()
        # Calculate probabilities for class 0
        probs_class0 = 1 - probs_class1
        # Stack the probabilities for both classes along the last axis
        return np.vstack((probs_class0, probs_class1)).T
