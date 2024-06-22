from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    def __init__(self, model_path, backend):
        """
        Initialize the BaseModel class
        
        Parameters:
        model: Pre-trained model
        """
        self.model_path = model_path
        self.backend = backend
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Input data variable and generate predictions
        
        Parameters:
        data: DataFrame type data variable
        
        Returns:
        DataFrame type prediction results
        """
        pass
