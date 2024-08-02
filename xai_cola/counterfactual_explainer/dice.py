import pandas as pd
import numpy as np

from .base_explainer import CounterFactualExplainer

from xai_cola.ml_model import Model
from xai_cola.data import PandasData

import dice_ml


FACTUAL_CLASS = 1
SHUFFLE_COUNTERFACTUAL = True

class DiCE(CounterFactualExplainer):

    def __init__(self, ml_model: Model, data: PandasData, sample_num):  # 根据需求增加输入变量
        super().__init__(ml_model, data, sample_num)
        
        self.factual, self.counterfactual = self.generate_counterfactuals()
    
    def get_factual_indices(self):
        x_factual_ext = self.x_factual_pandas.copy()
        predict = self.ml_model.predict(self.x_factual_pandas)
        x_factual_ext[self.target_name] = predict
        sampling_weights = np.exp(x_factual_ext[self.target_name].values.clip(min=0) * 4)
        indices = (x_factual_ext.sample(self.sample_num, weights=sampling_weights)).index
        return indices, x_factual_ext

    def generate_counterfactuals(self) -> pd.DataFrame:
        
        # DICE counterfactual generation logic
        indices, x_factual_ext = self.get_factual_indices()
        x_chosen = self.x_factual_pandas.loc[indices]
        # y_factual = self.ml_model.predict(x_chosen)

        # Prepare for DiCE
        dice_model = dice_ml.Model(model=self.ml_model, backend=self.ml_model.backend) #'sklearn'
        dice_features = x_chosen.columns.to_list()   #exclude 'Risk'
        dice_data = dice_ml.Data(
            dataframe = x_factual_ext,             # x_factual with 'Risk'
            continuous_features = dice_features,   # exclude 'Risk'
            outcome_name =self.target_name,              # 'Risk'
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
