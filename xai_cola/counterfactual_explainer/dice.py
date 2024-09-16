import pandas as pd
import numpy as np

from .base_explainer import CounterFactualExplainer

from xai_cola.ml_model import Model
from xai_cola.data import BaseData

import dice_ml


FACTUAL_CLASS = 1
SHUFFLE_COUNTERFACTUAL = True

class DiCE(CounterFactualExplainer):

    def __init__(self, ml_model: Model, data:BaseData=None):
        super().__init__(ml_model, data)
    

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

            ) -> np.ndarray:

        """
        Generate counterfactuals for the given factual

        Parameters:
        data: BaseData type, the factual data(don't need target column)
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

        # Prepare for DiCE
        dice_model = dice_ml.Model(model=self.ml_model, backend=self.ml_model.backend) #'sklearn'
        dice_features = x_chosen.columns.to_list() 
        dice_data = dice_ml.Data(
            dataframe = x_chosen,                 # factual, pd.DataFrame, without target column
            continuous_features = dice_features, 
            outcome_name =self.target_name,   
        )
        dice_explainer = dice_ml.Dice(dice_data, dice_model)
        dice_results = dice_explainer.generate_counterfactuals(
            query_instances = x_chosen,
            features_to_vary = dice_features,
            desired_class=1 - FACTUAL_CLASS,
            total_CFs=1,
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
    
    