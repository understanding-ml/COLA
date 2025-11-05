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

    def __init__(self, ml_model):
        """
        Initialize DiCE explainer

        Parameters:
        -----------
        ml_model : Model
            The machine learning model wrapper
            Should be created with Model(model=..., backend="sklearn")
            Can wrap:
            - Plain sklearn model (not recommended, preprocessing needed)
            - sklearn Pipeline with preprocessing (recommended)

        Example:
        --------
        # Create pipeline with preprocessing
        pipe = Pipeline([("pre", column_transformer), ("clf", lgbm_clf)])
        ml_model = Model(model=pipe, backend="sklearn")

        # Initialize DiCE (no data needed)
        dice = DiCE(ml_model=ml_model)

        # Generate counterfactuals (provide data here)
        factual_df, cf_df = dice.generate_counterfactuals(
            data=data,
            continuous_features=['age', 'income'],
            total_cfs=5
        )
        """
        super().__init__(ml_model)
    

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
            data: COLAData,
            factual_class: int = 1,
            total_cfs: int = 1,
            features_to_keep: list = None,
            continuous_features: list = None
            ) -> tuple:

        """
        Generate counterfactuals for the given factual data

        Parameters:
        -----------
        data : COLAData (required)
            Data wrapper containing the factual data (original raw data)
            Must include both features and target column
        factual_class : int, default=1
            The class of the factual data
            Normally, we set the factual_class as 1 (positive class)
            and we hope the counterfactual is 0 (negative class)
        total_cfs : int, default=1
            Total number of counterfactuals required for each query instance
        features_to_keep : list, optional
            List of feature names to keep unchanged in the counterfactuals
            Uses original feature names (before any preprocessing)
        continuous_features : list, optional
            List of continuous/numerical feature names for dice_ml.Data
            Uses original feature names (before any preprocessing)
            If None, will use all features as continuous features
            Categorical features are automatically inferred as all features minus continuous_features

        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Contains the original factual data
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values are set to (1 - factual_class)

        Raises:
        -------
        ValueError
            If data is None

        Example:
        --------
        # Prepare data
        data = COLAData(factual_data=X_raw, label_column='target',
                       numerical_features=['age', 'income'])

        # Generate counterfactuals
        factual_df, cf_df = dice.generate_counterfactuals(
            data=data,
            continuous_features=data.get_numerical_features(),
            features_to_keep=['gender'],  # Keep gender unchanged
            total_cfs=5
        )
        """

        # Call the data processing logic from the parent class
        self._process_data(data)

        # Get factual data (original raw data)
        x_factual = self.x_factual_pandas

        # Get predictions using ml_model (handles preprocessing internally if pipeline)
        pred_values = self.ml_model.predict(x_factual)

        # Create DataFrame with target column for DiCE
        x_with_targetcolumn = x_factual.copy()
        x_with_targetcolumn[self.target_name] = pred_values

        # Prepare for DiCE - use pipeline if available, otherwise use the model directly
        if self.ml_model.is_pipeline:
            # Use the full pipeline for DiCE (recommended)
            dice_model = dice_ml.Model(model=self.ml_model.pipeline, backend="sklearn")
        else:
            # Use the model directly (may not work well if preprocessing is needed)
            dice_model = dice_ml.Model(model=self.ml_model.model, backend=self.ml_model.backend)

        # Get original feature names (without target column)
        dice_features = x_factual.columns.to_list()

        # Determine which features to vary
        if features_to_keep is not None:
            features_to_vary = [feature for feature in dice_features if feature not in features_to_keep]
        else:
            features_to_vary = dice_features  # Default: allow all features to vary

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
            query_instances = x_factual,  # factual data (original raw data), pd.DataFrame, without target column
            features_to_vary = features_to_vary,
            desired_class="opposite",  # desired class is opposite of factual_class
            total_CFs=total_cfs,
        )

        # Iterate through each result and append to the DataFrame
        dice_df_list = []
        for cf in dice_results.cf_examples_list:
            # Convert to DataFrame and append
            cf_df = cf.final_cfs_df
            dice_df_list.append(cf_df)

        # Get counterfactual results (already in original feature space thanks to Pipeline)
        df_counterfactual = (
            pd.concat(dice_df_list).reset_index(drop=True).drop(self.target_name, axis=1)
        )
            
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
    
    