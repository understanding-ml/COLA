import pandas as pd
import numpy as np

from .pytorch_model import PyTorchModel
from .sklearn_model import SklearnModel 

class Model():
    """Classwrapper for pre-trained models."""
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend

        if self.backend not in ['pytorch', 'sklearn']:
            raise ValueError("Framework must be one of 'pytorch' and 'sklearn'")

        """Initialize the Model class
            
        :param model: Pre-trained model
        :param backend: 'pytorch', 'sklearn'.

                Attentiion! -> Our package will wrapper this part especially for DiCE: 
                Ues "TF1" ("TF2") for TensorFLow 1.0 (2.0), 
                "PYT" for PyTorch implementations, 
                "sklearn" for Scikit-Learn implementations of standard.
        """

    def predict(self, x_factual):
        if self.backend == "sklearn":
            return SklearnModel(self.model, self.backend).predict(x_factual)
        elif self.backend == "pytorch":
            return PyTorchModel(self.model, self.backend).predict(x_factual)
    
    def predict_proba(self, X):
        if self.backend == "sklearn":
            return SklearnModel(self.model, self.backend).predict_proba(X)
        elif self.backend == "pytorch":
            return PyTorchModel(self.model, self.backend).predict_proba(X)

    def to(self, device):
        if self.backend == "pytorch":
            return PyTorchModel(self.model, self.backend).to(device)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")

    def __call__(self, x):
        if self.backend == "pytorch":
            return PyTorchModel(self.model, self.backend)(x)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")
