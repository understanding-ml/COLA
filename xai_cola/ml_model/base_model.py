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


    def predict(self, data):
        """
        Generate predictions using the pre-trained model
        """
        pass

    def predict_proba(self, data):
        """
        Predict probability function that returns the probability distribution for each class.
        """
        pass
