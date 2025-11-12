from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    def __init__(self, model, backend):
        """
        Initialize the BaseModel class
        
        Parameters:
        model: Pre-trained model
        """
        self.model = model
        self.backend = backend.lower()
        if self.backend not in ['pytorch', 'sklearn', 'tensorflow1', 'tensorflow2']:
                raise ValueError("Framework must be one of 'pytorch', 'sklearn', 'tensorflow1', 'tensorflow2'")


    @abstractmethod
    def predict(self, x_factual):
        """
        Generate predictions using the pre-trained model.
        
        Parameters:
        -----------
        x_factual : np.ndarray or pd.DataFrame
            Input data for prediction
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        pass

    @abstractmethod
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
            Probability distribution for each class
        """
        pass
