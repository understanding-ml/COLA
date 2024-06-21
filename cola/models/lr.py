import torch.nn as nn
import torch
import numpy as np


class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

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
