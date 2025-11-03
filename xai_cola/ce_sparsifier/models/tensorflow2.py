import pandas as pd
import numpy as np
from .base import BaseModel

try:
    import tensorflow as tf
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False


class TensorFlow2Model(BaseModel):
    """Class wrapper for pre-trained TensorFlow 2.x models."""
    def __init__(self, model, backend="tensorflow2"):
        """
        Initialize the TensorFlow 2.x Model class
        
        Parameters:
        -----------
        model : tf.keras.Model or callable
            Pre-trained TensorFlow 2.x model
            - If tf.keras.Model: will use model.predict() method
            - If callable: should be a function that takes numpy array and returns predictions
        backend : str
            'tensorflow2' or 'tf2'
        """
        if not TF2_AVAILABLE:
            raise ImportError(
                "TensorFlow 2.x is not available. "
                "Please install tensorflow with: pip install tensorflow>=2.0"
            )
        super().__init__(model, backend)
        
        # Check if model is a Keras model
        if isinstance(model, tf.keras.Model):
            self.is_keras_model = True
        else:
            self.is_keras_model = False

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
        
        # If it's a Keras model, use predict method
        if self.is_keras_model:
            predictions = self.model.predict(X, verbose=0)
            # For binary classification, apply threshold
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                # Multi-class: use argmax
                predictions = np.argmax(predictions, axis=1)
            else:
                # Binary classification: use threshold
                predictions = (predictions.flatten() > 0.5).astype(int)
            return predictions
        
        # If model is callable
        if callable(self.model):
            predictions = self.model(X)
            # Convert tf.Tensor to numpy if needed
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            predictions = np.array(predictions)
            
            # Apply threshold for binary classification
            if predictions.ndim > 1:
                if predictions.shape[1] > 1:
                    predictions = np.argmax(predictions, axis=1)
                else:
                    predictions = (predictions.flatten() > 0.5).astype(int)
            else:
                predictions = (predictions > 0.5).astype(int)
            
            return predictions.flatten()
        
        raise ValueError(
            "Model must be a tf.keras.Model or a callable function that accepts numpy arrays."
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
        
        # If it's a Keras model, use predict method
        if self.is_keras_model:
            probs = self.model.predict(X, verbose=0)
            # For binary classification
            if probs.ndim > 1:
                if probs.shape[1] > 1:
                    # Multi-class: return probability of class 1 (index 1)
                    probs_class1 = probs[:, 1] if probs.shape[1] > 1 else probs.flatten()
                else:
                    # Binary classification: output is already probability of positive class
                    probs_class1 = probs.flatten()
            else:
                probs_class1 = probs.flatten()
            
            # Clamp to [0, 1]
            return np.clip(probs_class1, 0.0, 1.0)
        
        # If model is callable
        if callable(self.model):
            probs = self.model(X)
            # Convert tf.Tensor to numpy if needed
            if isinstance(probs, tf.Tensor):
                probs = probs.numpy()
            probs = np.array(probs)
            
            # Extract probability of class 1
            if probs.ndim > 1:
                if probs.shape[1] > 1:
                    probs_class1 = probs[:, 1]
                else:
                    probs_class1 = probs.flatten()
            else:
                probs_class1 = probs.flatten()
            
            # Clamp to [0, 1]
            return np.clip(probs_class1, 0.0, 1.0)
        
        raise ValueError(
            "Model must be a tf.keras.Model or a callable function that accepts numpy arrays."
        )

