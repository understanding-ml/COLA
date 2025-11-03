import pandas as pd
import numpy as np

from .pytorch import PyTorchModel
from .sklearn import SklearnModel
from .tensorflow1 import TensorFlow1Model
from .tensorflow2 import TensorFlow2Model 

class Model():
    """Class wrapper for pre-trained models."""
    def __init__(self, model, backend):
        """
        Initialize the Model class
        
        Parameters:
        -----------
        model : object
            Pre-trained model (sklearn, PyTorch, TensorFlow 1.x, or TensorFlow 2.x model)
        backend : str
            'pytorch', 'sklearn', 'tensorflow1'/'tf1', or 'tensorflow2'/'tf2'
            
        Note:
        -----
        For DiCE compatibility:
        - Use "TF1" ("TF2") for TensorFLow 1.0 (2.0)
        - Use "PYT" for PyTorch implementations
        - Use "sklearn" for Scikit-Learn implementations
        """
        self.model = model
        self.backend = backend.lower()
        
        # Normalize backend names (accept aliases)
        backend_map = {
            'tf1': 'tensorflow1',
            'tf2': 'tensorflow2',
            'tensorflow': 'tensorflow2',  # Default to TF2
        }
        if self.backend in backend_map:
            self.backend = backend_map[self.backend]

        if self.backend not in ['pytorch', 'sklearn', 'tensorflow1', 'tensorflow2']:
            raise ValueError(
                "Framework must be one of 'pytorch', 'sklearn', 'tensorflow1'/'tf1', or 'tensorflow2'/'tf2'"
            )

        # Cache model wrapper instances to avoid recreating them
        self._sklearn_model = None
        self._pytorch_model = None
        self._tensorflow1_model = None
        self._tensorflow2_model = None
    
    def _get_wrapped_model(self):
        """Get the appropriate wrapped model instance (cached)."""
        if self.backend == "sklearn":
            if self._sklearn_model is None:
                self._sklearn_model = SklearnModel(self.model, self.backend)
            return self._sklearn_model
        elif self.backend == "pytorch":
            if self._pytorch_model is None:
                self._pytorch_model = PyTorchModel(self.model, self.backend)
            return self._pytorch_model
        elif self.backend == "tensorflow1":
            if self._tensorflow1_model is None:
                self._tensorflow1_model = TensorFlow1Model(self.model, self.backend)
            return self._tensorflow1_model
        elif self.backend == "tensorflow2":
            if self._tensorflow2_model is None:
                self._tensorflow2_model = TensorFlow2Model(self.model, self.backend)
            return self._tensorflow2_model

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
        wrapped_model = self._get_wrapped_model()
        return wrapped_model.predict(x_factual)
    
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
            Probability of positive class (class 1) for binary classification
        """
        wrapped_model = self._get_wrapped_model()
        return wrapped_model.predict_proba(X)

    def to(self, device):
        """
        Move PyTorch model to specified device.
        
        Parameters:
        -----------
        device : str or torch.device
            Device to move model to (e.g., 'cuda', 'cpu')
        
        Returns:
        --------
        Model
            Returns self for method chaining
        
        Raises:
        -------
        NotImplementedError
            If backend is not 'pytorch'
        """
        if self.backend == "pytorch":
            wrapped_model = self._get_wrapped_model()
            wrapped_model.to(device)
            return self
        else:
            raise NotImplementedError("to() function can only be used for PyTorch models")

    def __call__(self, x):
        """
        Call the model directly (mainly for PyTorch models).
        
        Parameters:
        -----------
        x : torch.Tensor or np.ndarray
            Input data
        
        Returns:
        --------
        torch.Tensor or np.ndarray
            Model output
        
        Raises:
        -------
        NotImplementedError
            If backend is not 'pytorch'
        """
        if self.backend == "pytorch":
            wrapped_model = self._get_wrapped_model()
            return wrapped_model(x)
        else:
            raise NotImplementedError("__call__() can only be used for PyTorch models")
