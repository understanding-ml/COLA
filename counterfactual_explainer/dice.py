"""
    The algorithm we used is DiCE, from: https://github.com/interpretml/DiCE

    Paper for reference: 
    Explaining machine learning classifiers through diverse counterfactual explanations, from: https://doi.org/10.1145/3351095.3372850

"""

import pandas as pd
import numpy as np
import dice_ml

from .base_explainer import CounterFactualExplainer
from xai_cola.ml_model_interface import Model
from xai_cola.data_interface import BaseData

SHUFFLE_COUNTERFACTUAL = False

class DiCE(CounterFactualExplainer):

    def __init__(self, ml_model: Model, data:BaseData=None):
        super().__init__(ml_model, data)
        if self.ml_model.backend == 'sklearn':
            self.ml_model.backend = 'sklearn'
        elif self.ml_model.backend == 'pytorch':
            self.ml_model.backend = 'PYT'
    

    """
    Since we no more use the sample_num right now, we can remove the method 'get_factual_indices()'
    We will generate all the input data as factuals and return the counterfactuals throught the generate_counterfactuals method
    """
    # def get_factual_indices(self):
    #     """
    #     1' select the factuals whose prediction equals 1(if only 0 and 1) and return the indices
    #     2' return the x_factual_ext, which is the factuals with the target column, and the dataframe type
    #     """

    #     x_factual_ext = self.x_factual_pandas.copy()
    #     prediction = self.ml_model.predict(self.x_factual_pandas)
    #     x_factual_ext[self.target_name] = prediction
    #     sampling_weights = np.exp(x_factual_ext[self.target_name].values.clip(min=0) * 4)
    #     indices = (x_factual_ext.sample(self.sample_num, weights=sampling_weights)).index
    #     return indices, x_factual_ext


    def generate_counterfactuals(
            self, 
            data:BaseData=None,
            factual_class:int=1,
            total_cfs:int=1,
            features_to_keep: list = None  # Add this parameter to specify columns not to modify
            ) -> np.ndarray:

        """
        Generate counterfactuals for the given factual

        Parameters:
        param data: BaseData type, the factual data(don't need target column)
        param factual_class: The class of the factual data(Normally, we set the factual_class as the value 1 as the prediction of factual data is 1. And we hope the prediction of counterfactual data is 0)
        param total_CFs: Total number of counterfactuals required(of each query_instance).
        param features_to_keep: List of features to keep unchanged in the counterfactuals
        params: parameters for specific counterfactual algorithm

        return:
        factual, counterfactual: ndarray type
        """

        # Call the data processing logic from the parent class
        self._process_data(data)

        # It's related to the get_factual_indices() method, which we don't need anymore
        # indices, x_factual_ext = self.get_factual_indices()
        # x_chosen = self.x_factual_pandas.loc[indices]
        x_chosen = self.x_factual_pandas
        x_with_targetcolumn = x_chosen.copy() 
        x_with_targetcolumn[self.target_name] = self.ml_model.predict(x_chosen.values)
 
        # Prepare for DiCE
        dice_model = dice_ml.Model(model=self.ml_model, backend=self.ml_model.backend) #'sklearn'
        dice_features = x_chosen.columns.to_list() 

        # Remove the features you want to keep unchanged
        if features_to_keep is not None:
            features_to_vary = [feature for feature in dice_features if feature not in features_to_keep]
        else:
            features_to_vary = dice_features  # Default: allow all features to vary if none are specified

        dice_data = dice_ml.Data(
            dataframe = x_with_targetcolumn,                 # factual, pd.DataFrame, with target column
            continuous_features = dice_features, 
            outcome_name =self.target_name,   
        )
        dice_explainer = dice_ml.Dice(dice_data, dice_model)
        dice_results = dice_explainer.generate_counterfactuals(
            query_instances = x_chosen,                     # factual, pd.DataFrame, without target column
            features_to_vary = features_to_vary,
            desired_class=1 - factual_class,               # desired class is 0
            total_CFs=total_cfs,
        )

        # Iterate through each result and append to the DataFrame
        dice_df_list = []
        for cf in dice_results.cf_examples_list:
            # Convert to DataFrame and append
            cf_df = cf.final_cfs_df
            dice_df_list.append(cf_df)

        df_counterfactual = (
            pd.concat(dice_df_list).reset_index(drop=True).drop(self.target_name, axis=1)
        )
        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        factual = x_chosen.values
        counterfactual = df_counterfactual.values
        return factual , counterfactual   # return x and r    
    
    