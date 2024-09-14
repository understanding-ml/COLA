from abc import ABC, abstractmethod
import pandas as pd

from xai_cola.ml_model import Model
from xai_cola.data import BaseData

class CounterFactualExplainer(ABC):
    def __init__(self, ml_model:Model, data:BaseData, sample_num):
        """
        Initialize the CounterFactualExplainer class
        
        Parameters:
        model: Pre-trained model
        """
        self.ml_model = ml_model
        self.data = data                                                    # pandas type with 'target' column
        self.sample_num = sample_num

        self.x_factual_pandas = data.get_x()                                # pandas type without 'target' column
        # self.x_factual_pandas = data.get_dataframe()
        self.target_name = data.get_target_name()
        
        self.factual = None
        self.counterfacutal = None
        # self.factual, self.counterfactual = self.generate_counterfactuals()  # numpy type without 'target' column
        
    @abstractmethod
    def generate_counterfactuals(self) -> pd.DataFrame:
        """
        Generate counterfactuals for the given data
        
        Parameters:
        data: DataFrame type data variable
        
        Returns:
        DataFrame type counterfactual results
        """
        pass
