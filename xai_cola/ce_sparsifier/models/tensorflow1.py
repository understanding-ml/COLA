import pandas as pd
import numpy as np
from .base import BaseModel

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    TF1_AVAILABLE = True
except ImportError:
    TF1_AVAILABLE = False


class TensorFlow1Model(BaseModel):
    """Class wrapper for pre-trained TensorFlow 1.x models."""
    def __init__(self, model, backend="tensorflow1"):
        """
        Initialize the TensorFlow 1.x Model class
        
        Parameters:
        -----------
        model : tensorflow.compat.v1.Session or callable
            Pre-trained TensorFlow 1.x model
            - If Session: must have input/output tensors accessible
            - If callable: should be a function that takes numpy array and returns predictions
        backend : str
            'tensorflow1' or 'tf1'
        """
        if not TF1_AVAILABLE:
            raise ImportError(
                "TensorFlow 1.x is not available. "
                "Please install tensorflow with: pip install tensorflow"
            )
        super().__init__(model, backend)
        self.session = None
        self.input_tensor = None
        self.output_tensor = None
        
        # Try to set up session if model is a Session object
        if isinstance(model, tf.Session):
            self.session = model
            # Try to get input/output tensors from graph
            self._setup_tensors()

    def _setup_tensors(self):
        """Try to automatically detect input and output tensors from the graph."""
        if self.session is None:
            return
        
        graph = self.session.graph
        # Look for common input/output tensor names
        try:
            # Common patterns: 'input', 'x', 'features', etc.
            input_names = [op.name for op in graph.get_operations() 
                          if 'input' in op.name.lower() or 'x' == op.name.lower()]
            output_names = [op.name for op in graph.get_operations() 
                           if 'output' in op.name.lower() or 'y' == op.name.lower() or 'logits' in op.name.lower()]
            
            if input_names and output_names:
                self.input_tensor = graph.get_tensor_by_name(input_names[0] + ':0')
                self.output_tensor = graph.get_tensor_by_name(output_names[0] + ':0')
        except Exception:
            # If auto-detection fails, user will need to provide input/output manually
            pass

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
        X = np.array(X, dtype=np.float32)
        
        # Priority 1: If we have a session with tensors, use it
        if self.session is not None and self.input_tensor is not None and self.output_tensor is not None:
            predictions = self.session.run(
                self.output_tensor,
                feed_dict={self.input_tensor: X}
            )
            # Convert to binary predictions
            return (predictions.flatten() > 0.5).astype(int)
        
        # Priority 2: If model is callable, use it directly
        if callable(self.model):
            predictions = self.model(X)
            # Handle TensorFlow Tensor
            if isinstance(predictions, tf.Tensor):
                # Need session to evaluate tensor
                if self.session is not None:
                    predictions = self.session.run(predictions)
                else:
                    raise ValueError(
                        "TensorFlow Tensor requires a Session. "
                        "Please provide a model wrapped with Session, "
                        "or a callable that returns numpy arrays."
                    )
            predictions = np.array(predictions)
            # Convert to binary predictions
            return (predictions.flatten() > 0.5).astype(int)
        
        raise ValueError(
            "Cannot determine how to use the TensorFlow 1.x model. "
            "Please provide:\n"
            "1. A Session with accessible input/output tensors, or\n"
            "2. A callable function that returns numpy arrays"
        )

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
        X = np.array(X, dtype=np.float32)
        
        # Priority 1: If we have a session with tensors, use it
        if self.session is not None and self.input_tensor is not None and self.output_tensor is not None:
            probs = self.session.run(
                self.output_tensor,
                feed_dict={self.input_tensor: X}
            )
            probs = np.array(probs).flatten()
            # Clamp to [0, 1]
            return np.clip(probs, 0.0, 1.0)
        
        # Priority 2: If model is callable, use it directly
        if callable(self.model):
            probs = self.model(X)
            # Handle TensorFlow Tensor
            if isinstance(probs, tf.Tensor):
                # Need session to evaluate tensor
                if self.session is not None:
                    probs = self.session.run(probs)
                else:
                    raise ValueError(
                        "TensorFlow Tensor requires a Session. "
                        "Please provide a model wrapped with Session, "
                        "or a callable that returns numpy arrays."
                    )
            probs = np.array(probs).flatten()
            # Clamp to [0, 1]
            return np.clip(probs, 0.0, 1.0)
        
        raise ValueError(
            "Cannot determine how to use the TensorFlow 1.x model. "
            "Please provide:\n"
            "1. A Session with accessible input/output tensors, or\n"
            "2. A callable function that returns numpy arrays"
        )
