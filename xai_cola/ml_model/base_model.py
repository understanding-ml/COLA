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
        self.backend = backend
        

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
