import pandas as pd
import numpy as np

from .base_model import BaseModel

PYTORCH_MODELS = [
    "PyTorchRBFNet",
    "PyTorchDNN",
    "PyTorchLogisticRegression",
    "PyTorchLinearSVM",
]

class Model(BaseModel):
    def __init__(self, model, backend):
        super().__init__(model, backend)

    def predict(self, x_factual) -> pd.DataFrame:
        """
        Generate predictions using the pre-trained model
        
        Returns:
        DataFrame type prediction results
        """
        predictions = self.model.predict(x_factual)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame(name='Prediction')
        elif isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=['Prediction'])
        print(f'---- predictions have been made----')
        return predictions
    
    def predict_proba(self, x_factual):
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            x_factual = np.array(x_factual)
        proba = self.model.predict_proba(x_factual)
        if proba.ndim == 1:
            # 如果返回的是一维数组，直接返回
            return proba
        else:
            # 否则返回第二个类别的概率
            return proba[:, 1]

    def to(self, device):
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model.to(device)
        else:
            raise NotImplementedError("to function can only be used for PYTORCH_MODELS")

    def __call__(self, x):
        if self.model.__class__.__name__ in PYTORCH_MODELS:
            return self.model(x)
        else:
            raise NotImplementedError("__call__ can only be used for PYTORCH_MODELS")
