import pandas as pd
import numpy as np
import joblib
import pickle
from .base_model import BaseModel

PYTORCH_MODELS = [
    "PyTorchRBFNet",
    "PyTorchDNN",
    "PyTorchLogisticRegression",
    "PyTorchLinearSVM",
]

class PreTrainedModel(BaseModel):
    def __init__(self, model_path, backend):
        super().__init__(model_path, backend)


    def load_model(self):
        """
        Load a pre-trained model from a file
        
        Parameters:
        model_path: Path to the model file (.pkl or .joblib)
        
        Returns:
        Loaded model
        """
        if self.model_path.endswith('.pkl'):
            with open(self.model_path, 'rb') as f:
                self.model = joblib.load(f)
                print(f'----{self.model_path} model has been loaded----')
        elif self.model_path.endswith('.joblib'):
            self.model = joblib.load(self.model_path)
            print(f'----{self.model_path} model has been loaded----')
        else:
            raise ValueError("Unsupported model file format. Please provide a .pkl or .joblib file.")
        return self.model

    def predict(self, x_factual) -> pd.DataFrame:
        """
        Generate predictions using the pre-trained model
        
        Returns:
        DataFrame type prediction results
        """
        # model = self.load_model()
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
        return self.model.predict_proba(x_factual)[:, 1]


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
