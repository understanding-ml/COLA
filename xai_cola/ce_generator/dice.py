"""
    The algorithm we used is DiCE, from: https://github.com/interpretml/DiCE

    Paper for reference: 
    Explaining machine learning classifiers through diverse counterfactual explanations, from: https://doi.org/10.1145/3351095.3372850

"""

import pandas as pd
import numpy as np
import dice_ml

from .base_explainer import CounterFactualExplainer
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_sparsifier.data import COLAData

SHUFFLE_COUNTERFACTUAL = False

class DiCE(CounterFactualExplainer):

    def __init__(self, ml_model, data:COLAData=None):
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
            data:COLAData=None,
            factual_class:int=1,
            total_cfs:int=1,
            features_to_keep: list = None,
            continuous_features: list = None
            ) -> tuple:

        """
        Generate counterfactuals for the given factual

        Parameters:
        -----------
        data : COLAData, optional
            BaseData type, the factual data(don't need target column)
        factual_class : int, default=1
            The class of the factual data(Normally, we set the factual_class as the value 1 
            as the prediction of factual data is 1. And we hope the prediction of counterfactual data is 0)
        total_cfs : int, default=1
            Total number of counterfactuals required (of each query_instance)
        features_to_keep : list, optional
            List of features to keep unchanged in the counterfactuals
        continuous_features : list, optional
            List of continuous feature names for dice_ml.Data
            If None, will use all features as continuous features
            Categorical features are automatically inferred as all features minus continuous_features

        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)
        """

        # Call the data processing logic from the parent class
        self._process_data(data)

        # It's related to the get_factual_indices() method, which we don't need anymore
        # indices, x_factual_ext = self.get_factual_indices()
        # x_chosen = self.x_factual_pandas.loc[indices]
        x_chosen = self.x_factual_pandas
        
        # Apply transformation if needed
        if self.data.transform_method is not None:
            x_chosen_transformed = self.data._transform(x_chosen)
        else:
            x_chosen_transformed = x_chosen
        
        # Get predictions on transformed data if needed
        if self.data.transform_method is not None:
            pred_values = self.ml_model.predict(x_chosen_transformed.values)
        else:
            pred_values = self.ml_model.predict(x_chosen.values)
        
        x_with_targetcolumn = x_chosen_transformed.copy() 
        x_with_targetcolumn[self.target_name] = pred_values
 
        # Prepare for DiCE
        dice_model = dice_ml.Model(model=self.ml_model, backend=self.ml_model.backend) #'sklearn'
        dice_features = x_chosen_transformed.columns.to_list() 

        # Remove the features you want to keep unchanged
        if features_to_keep is not None:
            # Map original features to keep to transformed features
            if self.data.transform_method is not None:
                # For transformed data, need to find which transformed columns correspond to original features_to_keep
                # This is complex, so we'll skip this for now and just use the feature names as-is
                features_to_vary = [feature for feature in dice_features if feature not in features_to_keep]
            else:
                features_to_vary = [feature for feature in dice_features if feature not in features_to_keep]
        else:
            features_to_vary = dice_features  # Default: allow all features to vary if none are specified

        # Prepare dice_ml.Data parameters
        # Use provided continuous_features or default to all features
        if continuous_features is None:
            continuous_features = dice_features
        
        # Automatically infer categorical_features as all features minus continuous_features
        categorical_features = [feat for feat in dice_features if feat not in continuous_features]
        
        dice_data_params = {
            'dataframe': x_with_targetcolumn,  # factual, pd.DataFrame, with target column
            'continuous_features': continuous_features,
            'outcome_name': self.target_name,
        }
        # Add categorical_features if there are any
        if categorical_features:
            dice_data_params['categorical_features'] = categorical_features
        
        dice_data = dice_ml.Data(**dice_data_params)
        dice_explainer = dice_ml.Dice(dice_data, dice_model)
        dice_results = dice_explainer.generate_counterfactuals(
            query_instances = x_chosen_transformed,  # factual, pd.DataFrame, without target column
            features_to_vary = features_to_vary,
            desired_class=1 - factual_class,  # desired class is 0
            total_CFs=total_cfs,
        )

        # Iterate through each result and append to the DataFrame
        dice_df_list = []
        for cf in dice_results.cf_examples_list:
            # Convert to DataFrame and append
            cf_df = cf.final_cfs_df
            dice_df_list.append(cf_df)

        df_counterfactual_transformed = (
            pd.concat(dice_df_list).reset_index(drop=True).drop(self.target_name, axis=1)
        )
        
        # Inverse transform if needed
        if self.data.transform_method is not None:
            df_counterfactual = self.data._inverse_transform(df_counterfactual_transformed)
        else:
            df_counterfactual = df_counterfactual_transformed
            
        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        # Prepare factual with target column - get directly from COLAData
        factual_df = self.data.get_factual_all()
        
        # Prepare counterfactual with target column
        # Counterfactual target should be the desired class (1 - factual_class)
        counterfactual_target_value = 1 - factual_class
        counterfactual_df = df_counterfactual.copy()
        counterfactual_df[self.target_name] = counterfactual_target_value
        
        # Ensure column order matches factual (target column at the end)
        all_columns = factual_df.columns.tolist()
        counterfactual_df = counterfactual_df[all_columns]
        
        # Return pandas DataFrames directly
        return factual_df, counterfactual_df    
    
    