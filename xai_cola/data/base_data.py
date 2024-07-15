from abc import ABC, abstractmethod
import pandas as pd

class BaseData(ABC):
    def __init__(self, data, target_name):
        """
        Initialize the BaseData class
        
        Parameters:
        data : we want to use it as the input data
        """
        self.data = data
        self.target_name = target_name

    def get_dataframe(self) -> pd.DataFrame:

        pass


    def get_x(self):

        pass


    def get_y(self):
        pass

    
    def get_target_name(self):
        pass

    def get_x_labels(self):
        pass