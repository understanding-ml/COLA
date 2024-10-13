import pandas as pd
import numpy as np

from .base_model import BaseModel

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
        
        Returns:
        Numpy type prediction results
        """
        if isinstance(x_factual, pd.DataFrame):
            x_factual = x_factual.values
        return self.model.predict(x_factual)
    
    def predict_proba(self, X):
        """
        Predict probability function that returns the probability distribution for each class.

        Args:
        X (numpy.ndarray or pandas.DataFrame): Input data for which to predict probabilities.

        Returns:
        numpy.ndarray: Array of shape (n_samples, n_classes) containing the predicted probabilities
                    for each class. Each row corresponds to a sample, and each column corresponds
                    to a class.
        """
        # if X is DataFrame，transfer to np.array
        # if isinstance(X, pd.DataFrame):
        #     X = X.values
        # # get the probability of each class for each sample
        # probabilities = self.model.predict_proba(X)

        # # check if it is a 1D array (usually occurs in binary classification, only return the probability of the positive class)        
        # if probabilities.ndim == 1:
        #     # expand to a 2D array, column 0 is the probability of class 0, column 1 is the probability of class 1
        #     probabilities = np.vstack([1 - probabilities, probabilities]).T
        # return probabilities

        """
        classwrapper.py use this return (But I dont know why)
        """
        return self.model.predict_proba(X)[:, 1]
