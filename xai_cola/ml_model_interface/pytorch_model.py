import pandas as pd
import numpy as np
import torch
from .base_model import BaseModel

class PyTorchModel(BaseModel):
    """Classwrapper for pre-trained Pytorch models."""
    def __init__(self, model, backend="pytorch"):
        super().__init__(model, backend)

        """Initialize the Model class
            
        :param model: Pytorch model(must have 'forward() method' & 'to() method')
            where 'forward()' method is used to make predictions
        :param backend: 'pytorch'
        """

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, X):
        x = np.array(X)
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

    def predict_proba(self, X):
        x = np.array(X)
        x = torch.FloatTensor(x)
        # Forward pass to get output probabilities for class 1
        probs_class1 = self(x).reshape(-1).detach().numpy()
        # Calculate probabilities for class 0
        probs_class0 = 1 - probs_class1
        # Stack the probabilities for both classes along the last axis      
        return np.vstack((probs_class0, probs_class1)).T[:, 1]  

    """
    def predict_proba(self, X):
        x = np.array(X)
        x = torch.FloatTensor(x)
        # Forward pass to get output probabilities for class 1
        probs_class1 = self(x).reshape(-1).detach().numpy()
        # Calculate probabilities for class 0
        probs_class0 = 1 - probs_class1
        # Stack the probabilities for both classes along the last axis
        prob = np.vstack((probs_class0, probs_class1)).T
        return prob[:, 1]
    """
    
    def to(self, device):
        return self.model.to(device)
    
    def __call__(self, x):
        return self.model(x)
