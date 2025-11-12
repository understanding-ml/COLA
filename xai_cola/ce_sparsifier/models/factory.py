import pandas as pd
import numpy as np

from .pytorch import PyTorchModel
from .sklearn import SklearnModel
from .tensorflow1 import TensorFlow1Model
from .tensorflow2 import TensorFlow2Model 

class Model():
    """Class wrapper for pre-trained models with optional preprocessing pipeline."""
    def __init__(self, model, backend, preprocessor=None):
        """
        Initialize the Model class

        Parameters:
        -----------
        model : object
            Pre-trained model (sklearn, PyTorch, TensorFlow 1.x, or TensorFlow 2.x model)
            Can also be a sklearn Pipeline object
        backend : str
            'pytorch', 'sklearn', 'tensorflow1'/'tf1', or 'tensorflow2'/'tf2'
        preprocessor : sklearn.compose.ColumnTransformer or sklearn.pipeline.Pipeline, optional
            Preprocessing pipeline to apply before model prediction
            If model is already a Pipeline, this parameter will be ignored

        Note:
        -----
        For DiCE compatibility:
        - Use "TF1" ("TF2") for TensorFLow 1.0 (2.0)
        - Use "PYT" for PyTorch implementations
        - Use "sklearn" for Scikit-Learn implementations

        Example:
        --------
        # Option 1: Model with separate preprocessor
        model = Model(model=lgbm_clf, backend="sklearn", preprocessor=column_transformer)

        # Option 2: Model with Pipeline (recommended for DiCE)
        pipe = Pipeline([("pre", column_transformer), ("clf", lgbm_clf)])
        model = Model(model=pipe, backend="sklearn")
        """
        # Check if model is a Pipeline
        self.is_pipeline = hasattr(model, 'steps') and hasattr(model, 'named_steps')

        if self.is_pipeline:
            # Extract the actual model from pipeline for internal use
            # Assume the last step is the model
            self.pipeline = model
            self.model = model.steps[-1][1]  # Get the classifier from last step
            self.preprocessor = None  # Pipeline handles preprocessing internally
        else:
            self.pipeline = None
            self.model = model
            self.preprocessor = preprocessor

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
            Input data for prediction (raw data)

        Returns:
        --------
        np.ndarray
            Predictions
        """
        # If model is a pipeline, use it directly (handles preprocessing + prediction)
        if self.is_pipeline:
            return self.pipeline.predict(x_factual)

        # If preprocessor is provided, apply it first
        if self.preprocessor is not None:
            x_factual = self.preprocessor.transform(x_factual)

        # Use the wrapped model for prediction
        wrapped_model = self._get_wrapped_model()
        return wrapped_model.predict(x_factual)
    
    def predict_proba(self, X):
        """
        Predict probability function that returns the probability distribution for each class.

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input data for which to predict probabilities (raw data)

        Returns:
        --------
        np.ndarray
            Probability of positive class (class 1) for binary classification
        """
        # If model is a pipeline, use it directly (handles preprocessing + prediction)
        if self.is_pipeline:
            return self.pipeline.predict_proba(X)

        # If preprocessor is provided, apply it first
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        # Use the wrapped model for prediction
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
