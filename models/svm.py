import torch.nn as nn
import torch
import numpy as np


class PyTorchLinearSVM(nn.Module):
    """Linear SVM Classifier"""

    def __init__(self, input_dim):
        super(PyTorchLinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

        self.name = "svm"

    def forward(self, x):
        out = self.fc(x)
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
        # Forward pass to get output probabilities for class 1
        probs_class1 = self(x).reshape(-1).detach().numpy()
        # Calculate probabilities for class 0
        probs_class0 = 1 - probs_class1
        # Stack the probabilities for both classes along the last axis
        return np.vstack((probs_class0, probs_class1)).T


def svm_loss(outputs, labels):
    """Hinge loss for SVM"""
    return torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))
