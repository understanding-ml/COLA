import pandas as pd
import numpy as np
import torch
from .base import BaseModel

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

    def predict(self, X):
        """
        Generate predictions using the pre-trained model.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input data for prediction
        
        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        x = np.array(X)
        x = torch.FloatTensor(x)
        
        # Forward pass through the network
        with torch.no_grad():
            output = self(x)
        
        # Check for NaN in the output and replace with a default value
        if torch.isnan(output).any():
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        # Apply decision threshold (0.5) for binary classification
        predictions = (output.reshape(-1) > 0.5).float().numpy()
        return predictions.astype(int)

    def predict_proba(self, X):
        """
        Predict probability function that returns the probability distribution for each class.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input data for which to predict probabilities
        
        Returns:
        --------
        np.ndarray
            Probability of class 1 (positive class) for binary classification.
            Shape: (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        x = np.array(X)
        x = torch.FloatTensor(x)
        
        # Forward pass to get output probabilities for class 1
        with torch.no_grad():
            output = self(x)
        
        # Check for NaN in the output and replace with a default value
        if torch.isnan(output).any():
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        # Get probability of class 1 (positive class)
        probs_class1 = output.reshape(-1).detach().numpy()
        # Clamp probabilities to [0, 1] range
        probs_class1 = np.clip(probs_class1, 0.0, 1.0)
        
        return probs_class1
    
    def to(self, device):
        return self.model.to(device)
    
    def __call__(self, x):
        return self.model(x)
