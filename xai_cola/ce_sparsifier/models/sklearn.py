import pandas as pd
import numpy as np

from .base import BaseModel

class SklearnModel(BaseModel):
    """Classwrapper for pre-trained models."""
    def __init__(self, model, backend):
        super().__init__(model, backend)
        """Initialize the Model class
            
        :param model: Sklearn model
        :param backend: 'sklearn'
        """

    def predict(self, x_factual):
        """
        Generate predictions using the pre-trained model

        Parameters:
        -----------
        x_factual : np.ndarray or pd.DataFrame
            Input data for prediction

        Returns:
        --------
        np.ndarray
            Prediction results
        """
        return self.model.predict(x_factual)
    
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
            Array of shape (n_samples,) containing the predicted probabilities
            for the positive class (class 1) in binary classification.
        """
        return self.model.predict_proba(X)[:, 1]
