import pandas as pd
from .base_data import BaseData

# data contains the x and y, including their labels
class PandasData(BaseData): 
    def __init__(self, data, target_name):
        super().__init__(data, target_name)
    
    def get_dataframe(self):
        """
        Return the DataFrame stored in the class
        """
        return self.data
    
    def get_x(self):
        """
        Return the DataFrame excluding the target column
        """
        x_factual  = self.data.drop(columns=[self.target_name]).copy()
        return x_factual
    
    def get_y(self):
        """
        Return the data of the target column
        """
        return self.data[self.target_name]
    
    def get_target_name(self):
        return self.target_name
    
    def get_x_labels(self):
        a = self.data.drop(columns=[self.target_name]).copy()
        return a.columns
    
