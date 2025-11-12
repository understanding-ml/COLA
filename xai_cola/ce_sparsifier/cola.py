"""
COLA - COunterfactual with Limited Actions

Main class for refining counterfactual explanations with action-limited constraints.
"""

import string
import os
import numpy as np
import pandas as pd
from typing import Optional, List
from IPython.display import clear_output

from .data import COLAData
from .models import Model

from .policies.matching import (
    CounterfactualOptimalTransportPolicy,
    CounterfactualSoftCEMPolicy,
    CounterfactualNearestNeighborMatchingPolicy,
    CounterfactualExactMatchingPolicy
)
from .policies.data_composer import DataComposer
from .policies.feature_attributor import PSHAP
from .visualization.heatmap import generate_binary_heatmap
from .visualization.heatmap_direction import generate_direction_heatmap
from .visualization.highlight_dataframe import highlight_changes_comparison, highlight_changes_final
from .visualization.stacked_bar import generate_stacked_bar_chart


class COLA:
    """
    COLA (COunterfactual with Limited Actions) - Main class for refining counterfactuals
    
    This class orchestrates the entire workflow of refining counterfactual explanations
    by limiting the number of feature changes.
    
    """
    
    def __init__(
        self,
        data: COLAData,
        ml_model: Model,
    ):
        """
        Initialize COLA with data and model.

        Parameters:
        -----------
        data : COLAData
            Data wrapper containing factual and counterfactual data.
            Must have both factual and counterfactual data set.
        ml_model : Model
            Machine learning model interface

        Raises:
        -------
        ValueError
            If data does not have counterfactual data set
        """
        self.data = data
        self.ml_model = ml_model
        self.random_state = None  # Will be set in set_policy()

        # Verify that data has factual data
        if data.factual_df is None:
            raise ValueError(
                "Data must contain factual data. "
                "Please initialize COLAData with factual_data."
            )
        
        # Verify that data has counterfactual data (required for COLA)
        if not data.has_counterfactual():
            raise ValueError(
                "Data must contain counterfactual data for COLA to work. "
                "Please use data.add_counterfactuals() to add counterfactual data first."
            )

        # Verify that only one preprocessor exists (either in data or in model, not both)
        if data.transform_method is not None and ml_model.is_pipeline:
            raise ValueError(
                "Cannot have preprocessor in both COLAData (transform_method) and Model (pipeline). "
                "Please choose one:\n"
                "  - Use COLAData with transform_method and a non-pipeline model, OR\n"
                "  - Use a pipeline model with preprocessing and set transform_method=None in COLAData"
            )
        
        # Extract factual and counterfactual from data
        # Store as DataFrames to preserve column names for sklearn Pipeline compatibility
        self.x_factual_pandas = data.get_factual_features()
        self.x_counterfactual_pandas = data.get_counterfactual_features()

        # Also store numpy versions for backward compatibility
        self.x_factual = data.to_numpy_factual_features()
        self.x_counterfactual = data.to_numpy_counterfactual_features()

        # Get transformed data from COLAData if available
        # COLAData已经在初始化时自动转换并保存了转换后的数据
        self.x_factual_transformed = None
        self.x_counterfactual_transformed = None
        self.has_transformed_data = False

        if data.has_transformed_data():
            # 直接从 COLAData 获取转换后的数据（DataFrame 格式）
            self.x_factual_transformed_df = data.get_transformed_factual_features()
            self.x_counterfactual_transformed_df = data.get_transformed_counterfactual_features()

            # 同时保存 NumPy 格式用于计算
            self.x_factual_transformed = data.to_numpy_transformed_factual_features()
            self.x_counterfactual_transformed = data.to_numpy_transformed_counterfactual_features()

            self.has_transformed_data = True
            print(f"Using transformed data from COLAData (transform_method: {type(data.transform_method).__name__})")

        self.row_indices = None
        self.col_indices = None
        
        # Initialize policy parameters
        self.matcher = None
        self.attributor = None
        self.Avalues_method = None
        
        # Flag to track if minimum actions can achieve target (used internally)
        self._is_feasible_with_features = True
    
    def set_policy(
        self,
        matcher: str = "ot",
        attributor: str = "pshap",
        random_state: int = 42,
        **kwargs
    ):
        """
        Set the refinement policy.

        Parameters:
        -----------
        matcher : str
            Matching strategy between factual and counterfactual
            Options: "ot" (Optimal Transport), "ect" (Exact Matching),
                     "nn" (Nearest Neighbor), "cem" (Coarsened Exact Matching)
        attributor : str
            Feature attribution method
            Options: "pshap" (PSHAP with joint probability)
        random_state : int, optional
            Random seed used to control the reproducibility of counterfactual actions sampling.
            The same random_state means the same action sequence is sampled whenever
            counterfactuals or minimum-actions are queried. Default is 42.
        **kwargs
            Additional parameters for matcher and attributor
        """
        self.matcher = matcher
        self.attributor = attributor
        self.random_state = random_state
        self.matcher_params = kwargs
        
        # Validate matcher
        matcher_names = {
            "ot": "Optimal Transport Matching",
            "cem": "Coarsened Exact Matching",
            "nn": "Nearest Neighbor Matching",
            "ect": "Exact Matching",
            "cem": "Coarsened Exact Matching with Optimal Transport"
        }
        
        if matcher not in matcher_names:
            raise ValueError(
                f"{matcher} is not a valid matcher, please choose from: {list(matcher_names.keys())}"
            )
        
        matcher_name = matcher_names[matcher]
        print(f"Policy set: {attributor} with {matcher_name}.")
    
    def get_refined_counterfactual(self, limited_actions: int, features_to_vary: Optional[List[str]] = None):
        """
        Get counterfactuals refined with limited actions.
        
        Parameters:
        -----------
        limited_actions : int
            Maximum number of feature changes to apply
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        pd.DataFrame
            Refined counterfactual DataFrame with target column
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
        _, _, refined_counterfactual_df = self._compute_refined_results(limited_actions, features_to_vary)
        return refined_counterfactual_df
    
    def get_all_results(self, limited_actions: int, features_to_vary: Optional[List[str]] = None):
        """
        Get all results: factual, counterfactual, and refined counterfactual.
        
        Parameters:
        -----------
        limited_actions : int
            Maximum number of feature changes to apply
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        tuple
            (factual_df, counterfactual_df, refined_counterfactual_df)
            All are pd.DataFrame with target column
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
        return self._compute_refined_results(limited_actions, features_to_vary)
    
    def _compute_refined_results(self, limited_actions: int, features_to_vary: Optional[List[str]] = None):
        """
        Internal method to compute refined counterfactual results.
        
        Parameters:
        -----------
        limited_actions : int
            Maximum number of feature changes to apply
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        tuple
            (factual_df, counterfactual_df, refined_counterfactual_df)
            All are pd.DataFrame with target column
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
        # Compute attribution and composition
        varphi = self._get_attributor()
        q_transformed = self._get_data_composer()  # q 在转换空间中（如果使用了转换）

        # 统一使用转换后的数据进行所有计算和修改
        # 只在最后展示时才转换回原始空间
        q = q_transformed  # q 始终在转换空间中（如果有转换的话）

        # Get allowed column indices if features_to_vary is specified
        allowed_col_indices = None
        if features_to_vary is not None:
            feature_columns = self.data.get_feature_columns()
            # Validate feature names
            invalid_features = [f for f in features_to_vary if f not in feature_columns]
            if invalid_features:
                raise ValueError(
                    f"The following features are not valid feature column names: {invalid_features}. "
                    f"Valid feature columns are: {feature_columns}"
                )
            # Get column indices for the specified features
            allowed_col_indices = [feature_columns.index(f) for f in features_to_vary]
            allowed_col_indices = np.array(allowed_col_indices)
        
        # Calculate minimum required actions if needed
        # If limited_actions is greater than or equal to minimum required, use minimum required
        # 使用转换后的数据进行模型预测（如果有转换的话）
        if self.has_transformed_data:
            # 直接使用转换后的数值数据
            y_corresponding_counterfactual = self.ml_model.predict(
                pd.DataFrame(q, columns=self.data.get_transformed_feature_columns())
            )
        else:
            y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(q))

        minimum_actions, is_feasible = self._compute_minimum_actions(
            varphi, q, y_corresponding_counterfactual, allowed_col_indices
        )
        self._is_feasible_with_features = is_feasible
        
        # Check if target is achievable with given features
        if not is_feasible and features_to_vary is not None:
            import warnings
            warnings.warn(
                f"Warning: The specified features_to_vary ({features_to_vary}) may not be sufficient "
                f"to achieve target predictions for all samples. "
                f"Using all available actions ({minimum_actions}) from the specified features, "
                f"but some samples may not reach the target prediction. "
                f"Consider including more features in features_to_vary.",
                UserWarning
            )
        
        # Determine actual actions to use
        if limited_actions >= minimum_actions:
            # Use minimum required actions for optimal result
            actual_actions = minimum_actions
        else:
            # Use limited_actions if it's less than minimum required
            actual_actions = limited_actions
        
        self.limited_actions = actual_actions
        
        # 生成完整随机采样序列
        # Get maximum possible actions (considering allowed_col_indices if specified)
        # If allowed_col_indices is specified, count actual different positions between factual and counterfactual
        # 注意：如果使用了转换，则 factual 和 q 都在转换空间中
        if self.has_transformed_data:
            x_factual_for_calc = self.x_factual_transformed
        else:
            x_factual_for_calc = self.x_factual

        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = x_factual_for_calc[:, allowed_col_indices]
            counterfactual_allowed = q[:, allowed_col_indices]
            diff_mask = factual_allowed != counterfactual_allowed
            total_actions = np.sum(diff_mask)
        else:
            total_actions = varphi.size

        all_row_indices, all_col_indices = self._get_action_sequence(varphi, total_actions, allowed_col_indices)

        # 取前 actual_actions 个
        row_indices = all_row_indices[:actual_actions]
        col_indices = all_col_indices[:actual_actions]
        self.row_indices = row_indices
        self.col_indices = col_indices

        # Apply selected actions（在转换空间中修改，如果使用了转换）
        q_values = q[row_indices, col_indices]
        x_action_constrained = x_factual_for_calc.copy()

        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val

        corresponding_counterfactual = q

        # DEBUG: 检查 factual 的预测
        if self.has_transformed_data:
            x_factual_df = pd.DataFrame(
                x_factual_for_calc,
                columns=self.data.get_transformed_feature_columns()
            )
            y_factual_pred = self.ml_model.predict(x_factual_df)
            print(f"\n[DEBUG] Predictions on transformed factual: {y_factual_pred}")

        # Get predictions（使用转换空间的数据进行预测）
        # NOTE: Use corresponding_counterfactual (q) which is the result of DataComposer
        # This ensures we have N counterfactuals corresponding to N factuals (even if original had M counterfactuals)
        if self.has_transformed_data:
            # DEBUG: 检查转换后的 counterfactual 预测
            x_counterfactual_transformed_df = pd.DataFrame(
                self.x_counterfactual_transformed,
                columns=self.data.get_transformed_feature_columns()
            )
            y_cf_transformed_pred = self.ml_model.predict(x_counterfactual_transformed_df)
            print(f"\n[DEBUG] Predictions on transformed counterfactual: {y_cf_transformed_pred}")

            # DEBUG: 检查 x_action_constrained 的预测
            x_action_constrained_df = pd.DataFrame(
                x_action_constrained,
                columns=self.data.get_transformed_feature_columns()
            )
            y_counterfactual_limited = self.ml_model.predict(x_action_constrained_df)
            print(f"[DEBUG] Predictions on x_action_constrained (before inverse_transform): {y_counterfactual_limited}")

            # DEBUG: 检查 corresponding_counterfactual (q) 的预测
            corresponding_counterfactual_df = pd.DataFrame(
                corresponding_counterfactual,
                columns=self.data.get_transformed_feature_columns()
            )
            y_corresponding_counterfactual = self.ml_model.predict(corresponding_counterfactual_df)
            print(f"[DEBUG] Predictions on corresponding_counterfactual (q): {y_corresponding_counterfactual}")
        else:
            y_counterfactual_limited = self.ml_model.predict(self._to_dataframe(x_action_constrained))
            y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(corresponding_counterfactual))

        # Convert to DataFrames
        # 如果使用了转换，需要将结果转换回原始空间进行展示
        if self.has_transformed_data:
            # 将转换空间的数据转换回原始空间
            corresponding_counterfactual_df = pd.DataFrame(
                corresponding_counterfactual,
                columns=self.data.get_transformed_feature_columns()
            )
            corresponding_counterfactual_original = self.data._inverse_transform(corresponding_counterfactual_df)
            # _inverse_transform 已经返回了正确列顺序的 DataFrame，不需要再重排

            x_action_constrained_df = pd.DataFrame(
                x_action_constrained,
                columns=self.data.get_transformed_feature_columns()
            )
            x_action_constrained_original = self.data._inverse_transform(x_action_constrained_df)
            # _inverse_transform 已经返回了正确列顺序的 DataFrame，不需要再重排
        else:
            corresponding_counterfactual_original = corresponding_counterfactual
            x_action_constrained_original = x_action_constrained

        # Use original labels from COLAData for factual_df instead of model predictions
        factual_df = self.data.get_factual_all()

        # 如果 inverse_transform 返回的是 DataFrame，转换为 NumPy array
        if isinstance(corresponding_counterfactual_original, pd.DataFrame):
            corresponding_counterfactual_original = corresponding_counterfactual_original.values
        if isinstance(x_action_constrained_original, pd.DataFrame):
            x_action_constrained_original = x_action_constrained_original.values

        # Use corresponding_counterfactual (q) instead of original x_counterfactual
        # This ensures counterfactual_df has the same number of rows as factual_df
        counterfactual_df = self._return_dataframe(
            corresponding_counterfactual_original, y_corresponding_counterfactual, factual_df.index
        )

        refined_counterfactual_df = self._return_dataframe(
            x_action_constrained_original, y_counterfactual_limited, factual_df.index
        )

        # Convert numerical features to int type for counterfactual_df and refined_counterfactual_df
        # This must be done BEFORE storing to self.ce_dataframe and self.ace_dataframe
        numerical_features = self.data.get_numerical_features() if hasattr(self.data, 'get_numerical_features') else []
        if numerical_features:
            for col in numerical_features:
                if col in counterfactual_df.columns:
                    try:
                        counterfactual_df[col] = counterfactual_df[col].round().astype(int)
                    except (ValueError, TypeError):
                        # If conversion fails, keep original dtype
                        pass
                if col in refined_counterfactual_df.columns:
                    try:
                        refined_counterfactual_df[col] = refined_counterfactual_df[col].round().astype(int)
                    except (ValueError, TypeError):
                        # If conversion fails, keep original dtype
                        pass

        # Store for backward compatibility with other methods (highlight_changes_comparison, highlight_changes_final, heatmap, etc.)
        self.factual_dataframe = factual_df
        self.ce_dataframe = counterfactual_df
        self.ace_dataframe = refined_counterfactual_df
        self.corresponding_counterfactual_dataframe = self._return_dataframe(
            corresponding_counterfactual_original, y_corresponding_counterfactual, factual_df.index
        )

        # Apply same data types to corresponding_counterfactual_dataframe
        # This ensures it has the same int types as counterfactual_df
        for col in counterfactual_df.columns:
            self.corresponding_counterfactual_dataframe[col] = \
                self.corresponding_counterfactual_dataframe[col].astype(
                    counterfactual_df[col].dtype
                )

        return factual_df, counterfactual_df, refined_counterfactual_df
    
    def highlight_changes_comparison(self):
        """
        Highlight changes in the dataframes with comparison format (old -> new).
        
        This is a thin wrapper around the pure visualization function. It retrieves
        pre-computed results from the COLA instance and passes them to the visualization
        function. Must call get_refined_counterfactual() or get_all_results() first.
        
        This method displays changes in the format "factual_value -> counterfactual_value"
        to show both the original and modified values side by side.
        
        Returns:
        --------
        tuple
            (factual_df, ce_style, ace_style)
            - factual_df: pandas.DataFrame - Original factual data
            - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → full counterfactual
            - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → action-limited counterfactual
        
        Usage:
        ------
        >>> factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()
        >>> # Display in Jupyter notebook
        >>> display(ce_style)
        >>> # Save as HTML
        >>> ce_style.to_html('changes.html')
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None or self.corresponding_counterfactual_dataframe is None:
            raise ValueError(
                "Cannot visualize changes: refined counterfactuals have not been generated yet. "
                "Please call get_refined_counterfactual() or get_all_results() method first before using visualization."
            )

        # Call pure visualization function
        return highlight_changes_comparison(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column()
        )
    
    def highlight_changes_final(self):
        """
        Highlight changes in the dataframes showing only the final values.
        
        This is a thin wrapper around the pure visualization function. It retrieves
        pre-computed results from the COLA instance and passes them to the visualization
        function. Must call get_refined_counterfactual() or get_all_results() first.
        
        This method displays only the final counterfactual values without showing
        the "factual -> counterfactual" format, making it cleaner for presentation.
        
        Returns:
        --------
        tuple
            (factual_df, ce_style, ace_style)
            - factual_df: pandas.DataFrame - Original factual data
            - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing only full counterfactual values
            - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing only action-limited counterfactual values
        
        Usage:
        ------
        >>> factual_df, ce_style, ace_style = refiner.highlight_changes_final()
        >>> # Display in Jupyter notebook
        >>> display(ce_style)
        >>> # Save as HTML
        >>> ce_style.to_html('changes.html')
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None or self.corresponding_counterfactual_dataframe is None:
            raise ValueError(
                "Cannot visualize changes: refined counterfactuals have not been generated yet. "
                "Please call get_refined_counterfactual() or get_all_results() method first before using visualization."
            )
        
        # Call pure visualization function
        return highlight_changes_final(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column()
        )
    
    def heatmap_binary(self, save_path: Optional[str] = None, save_mode: str = "combined", show_axis_labels: bool = True):
        """
        Generate binary change heatmap visualizations (shows if value changed or not).
        
        This is a thin wrapper around the pure visualization function. It retrieves
        pre-computed results from the COLA instance and passes them to the visualization
        function. Must call get_refined_counterfactual() or get_all_results() first.
        
        This method generates heatmaps showing binary changes: whether a value 
        changed (red) or remained unchanged (lightgrey).
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the heatmap images. 
            If None, plots are automatically displayed in Jupyter (default).
            If provided, saves to the specified path and closes the plots.
            Can be a directory path or file path.
            
        save_mode : str, default="combined"
            How to save the heatmaps when save_path is provided:
            - "combined": Save both heatmaps in a single combined image (top and bottom)
            - "separate": Save two separate image files (heatmap_ce.png and heatmap_ace.png)
            - Ignored if save_path is None
        show_axis_labels : bool, default=True
            Whether to show x and y axis labels (column names and row indices).
            If True, displays column names and row indices. If False, hides them.
            
        Returns:
        --------
        tuple
            (plot1, plot2) - Heatmap plots (matplotlib Figure objects)
        
        Examples:
        ---------
        >>> # Display plots in Jupyter (no saving)
        >>> refiner.heatmap_binary()
        >>> # Plots are automatically displayed in Jupyter notebook
        
        >>> # Save as combined image
        >>> refiner.heatmap_binary(save_path='./results', save_mode='combined')
        >>> # Creates: ./results/combined_heatmap.png (two heatmaps stacked vertically)
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None or self.corresponding_counterfactual_dataframe is None:
            raise ValueError(
                "Cannot visualize changes: refined counterfactuals have not been generated yet. "
                "Please call get_refined_counterfactual() or get_all_results() method first before using visualization."
            )
        
        # Call pure visualization function
        return generate_binary_heatmap(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column(),
            save_path=save_path,
            save_mode=save_mode,
            show_axis_labels=show_axis_labels
        )
    
    def heatmap_direction(self, save_path: Optional[str] = None, save_mode: str = "combined", show_axis_labels: bool = True):
        """
        Generate directional change heatmap visualizations (shows if value increased, decreased, or unchanged).

        This is a thin wrapper around the pure visualization function. It retrieves
        pre-computed results from the COLA instance and passes them to the visualization
        function. Must call get_refined_counterfactual() or get_all_results() first.

        This method generates heatmaps showing the direction of changes:
        - Numerical features (increased): light blue
        - Numerical features (decreased): light cyan
        - Categorical features (changed): peru (soft warm tone)
        - Value unchanged: lightgrey
        - Target column: changed values shown in black
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the heatmap images. 
            If None, plots are automatically displayed in Jupyter (default).
            If provided, saves to the specified path and closes the plots.
            Can be a directory path or file path.
            
        save_mode : str, default="combined"
            How to save the heatmaps when save_path is provided:
            - "combined": Save both heatmaps in a single combined image (top and bottom)
            - "separate": Save two separate image files
            - Ignored if save_path is None
        show_axis_labels : bool, default=True
            Whether to show x and y axis labels (column names and row indices).
            If True, displays column names and row indices. If False, hides them.
            
        Returns:
        --------
        tuple
            (plot1, plot2) - Heatmap plots (matplotlib Figure objects)
        
        Examples:
        ---------
        >>> # Display plots in Jupyter (no saving)
        >>> refiner.heatmap_direction()
        >>> # Plots are automatically displayed in Jupyter notebook
        
        >>> # Save as combined image
        >>> refiner.heatmap_direction(save_path='./results', save_mode='combined')
        >>> # Creates: ./results/combined_direction_heatmap.png
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None or self.corresponding_counterfactual_dataframe is None:
            raise ValueError(
                "Cannot visualize changes: refined counterfactuals have not been generated yet. "
                "Please call get_refined_counterfactual() or get_all_results() method first before using visualization."
            )
        
        # Get numerical features from data object
        numerical_features = self.data.get_numerical_features() if hasattr(self.data, 'get_numerical_features') else None

        # Call pure visualization function
        return generate_direction_heatmap(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column(),
            numerical_features=numerical_features,
            save_path=save_path,
            save_mode=save_mode,
            show_axis_labels=show_axis_labels
        )
    
    def query_minimum_actions(self, features_to_vary: Optional[List[str]] = None):
        """
        Find the minimum number of actions needed.
        
        Parameters:
        -----------
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        int
            Minimum number of actions
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
        self.varphi = self._get_attributor()
        self.q = self._get_data_composer()
        
        # Get allowed column indices if features_to_vary is specified
        allowed_col_indices = None
        if features_to_vary is not None:
            feature_columns = self.data.get_feature_columns()
            # Validate feature names
            invalid_features = [f for f in features_to_vary if f not in feature_columns]
            if invalid_features:
                raise ValueError(
                    f"The following features are not valid feature column names: {invalid_features}. "
                    f"Valid feature columns are: {feature_columns}"
                )
            # Get column indices for the specified features
            allowed_col_indices = [feature_columns.index(f) for f in features_to_vary]
            allowed_col_indices = np.array(allowed_col_indices)

        # Use transformed data for model prediction if available (consistent with _compute_refined_results)
        if self.has_transformed_data:
            # Directly use transformed numerical data
            y_corresponding_counterfactual = self.ml_model.predict(
                pd.DataFrame(self.q, columns=self.data.get_transformed_feature_columns())
            )
        else:
            y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(self.q))

        minimum_actions, is_feasible = self._compute_minimum_actions(
            self.varphi, self.q, y_corresponding_counterfactual, allowed_col_indices
        )
        
        clear_output(wait=True)
        
        if not is_feasible:
            print(f"Warning: The specified features_to_vary may not be sufficient to achieve target predictions for all samples.")
            print(f"Using all available actions ({minimum_actions}) from the specified features.")
            print(f"Some samples may not reach the target prediction. Consider including more features.")
        else:
            print(f"The minimum number of actions is {minimum_actions}")

        return minimum_actions
    
    def stacked_bar_chart(self, save_path: Optional[str] = None, refined_color: str = '#D9F2D0', counterfactual_color: str = '#FBE3D6', instance_labels: Optional[List[str]] = None):
        """
        Generate a horizontal stacked percentage bar chart comparing modification positions.

        This is a thin wrapper around the pure visualization function. It retrieves
        pre-computed results from the COLA instance and passes them to the visualization
        function. Must call get_refined_counterfactual() or get_all_results() first.

        This method creates a percentage-based stacked bar chart where each bar represents
        an instance (100% total), showing the proportion of modified positions in refined
        counterfactual vs. original counterfactual relative to factual data.

        Each bar shows:
        - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
        - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
        - Total bar length: 100% (representing all counterfactual modifications)

        Labels on bars show both percentage and actual count (e.g., "60.0% (3)")

        Parameters:
        -----------
        save_path : str, optional
            Path to save the chart image. If None, chart is not saved.
            Can be a directory path or file path.
        refined_color : str, default='#D9F2D0'
            Color for refined counterfactual modified positions (light green)
        counterfactual_color : str, default='#FBE3D6'
            Color for counterfactual modified positions (light pink/orange)
        instance_labels : list, optional
            Custom labels for instances. If None, uses "instance 1", "instance 2", etc.
            Length must match the number of instances.

        Returns:
        --------
        matplotlib.figure.Figure
            The stacked bar chart figure

        Examples:
        ---------
        >>> # Display chart in Jupyter (no saving)
        >>> fig = refiner.stacked_bar_chart()
        >>> # Chart is automatically displayed in Jupyter notebook

        >>> # Save chart to file
        >>> fig = refiner.stacked_bar_chart(save_path='./results')
        >>> # Creates: ./results/stacked_bar_chart.png

        >>> # Custom instance labels and colors
        >>> fig = refiner.stacked_bar_chart(
        ...     instance_labels=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'],
        ...     refined_color='#D9F2D0',
        ...     counterfactual_color='#FBE3D6'
        ... )

        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet.
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None or self.corresponding_counterfactual_dataframe is None:
            raise ValueError(
                "Cannot visualize changes: refined counterfactuals have not been generated yet. "
                "Please call get_refined_counterfactual() or get_all_results() method first before using visualization."
            )
        
        # Validate instance_labels if provided
        n_instances = len(self.factual_dataframe)
        if instance_labels is not None and len(instance_labels) != n_instances:
            raise ValueError(
                f"Number of instance labels ({len(instance_labels)}) must match "
                f"number of instances ({n_instances})"
            )
        
        # Call pure visualization function
        return generate_stacked_bar_chart(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column(),
            save_path=save_path,
            refined_color=refined_color,
            counterfactual_color=counterfactual_color,
            instance_labels=instance_labels
        )

    def diversity(self):
        """
        Generate diversity analysis showing minimal feature combinations that can flip the target.

        This method finds all minimal feature combinations that can change the prediction
        from the factual's target value to the refined counterfactual's target value (e.g., from 1 to 0).
        For each instance, it returns styled DataFrames showing the different minimal combinations.

        The algorithm:
        1. For each instance, identify features that differ between factual and refined counterfactual
        2. Test combinations of increasing size (1 feature, 2 features, etc.)
        3. Find minimal sets that flip the prediction from factual target to refined counterfactual target
        4. Skip larger combinations that contain successful smaller combinations

        Returns:
        --------
        Tuple[pd.DataFrame, List[Styler]]
            - factual_df: Original factual data (copy)
            - diversity_styles: List of styled DataFrames (one per instance),
              each showing all minimal feature combinations for that instance

        Example:
        --------
        >>> factual_df, diversity_styles = sparsifier.diversity()
        >>> # Display results for each instance
        >>> for i, style in enumerate(diversity_styles):
        >>>     print(f"Instance {i+1} diversity:")
        >>>     display(style)
        """
        from .visualization import generate_diversity_for_all_instances

        # Check if refined counterfactuals have been generated
        if self.factual_dataframe is None or self.ace_dataframe is None:
            raise ValueError(
                "Cannot perform diversity analysis: refined counterfactuals have not been generated yet. "
                "Please run get_refined_counterfactual() or get_all_results() first."
            )

        # Check if ml_model is available
        if self.ml_model is None:
            raise ValueError(
                "Cannot perform diversity analysis: ML model is not available. "
                "Please provide ml_model when creating the sparsifier."
            )

        # Call pure visualization function
        return generate_diversity_for_all_instances(
            factual_df=self.factual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            ml_model=self.ml_model,
            label_column=self.data.get_label_column(),
            cola_data=self.data
        )

    # ========== Private Methods ==========

    # def _get_matcher(self):
    #     """Get matcher based on policy."""
    #     # For matchers that require numerical distance computation (ot, cem, nn),
    #     # we need to preprocess the data using the pipeline's preprocessor
    #     # to convert categorical features (strings) to numerical representations
    #     needs_preprocessing = self.matcher in ["ot", "cem", "nn"]

    #     if needs_preprocessing and self.ml_model.is_pipeline:
    #         # Extract preprocessor from pipeline
    #         try:
    #             import scipy.sparse as sp
    #             preprocessor = self.ml_model.model.named_steps.get('preprocessor')
    #             if preprocessor is not None:
    #                 # Transform data using preprocessor (OneHotEncoder returns sparse matrix by default)
    #                 x_factual_transformed = preprocessor.transform(self.x_factual_pandas)
    #                 x_counterfactual_transformed = preprocessor.transform(self.x_counterfactual_pandas)

    #                 # Convert sparse matrix to dense array (OT requires dense arrays)
    #                 if sp.issparse(x_factual_transformed):
    #                     x_factual_transformed = x_factual_transformed.toarray()
    #                 if sp.issparse(x_counterfactual_transformed):
    #                     x_counterfactual_transformed = x_counterfactual_transformed.toarray()

    #                 # Ensure float type for numerical operations
    #                 x_factual_transformed = np.asarray(x_factual_transformed, dtype=float)
    #                 x_counterfactual_transformed = np.asarray(x_counterfactual_transformed, dtype=float)
    #             else:
    #                 # No preprocessor found, use original data
    #                 x_factual_transformed = self.x_factual
    #                 x_counterfactual_transformed = self.x_counterfactual
    #         except Exception as e:
    #             # If preprocessing fails, fall back to original data and print warning
    #             print(f"Warning: Preprocessing failed ({type(e).__name__}: {e}), using original data")
    #             x_factual_transformed = self.x_factual
    #             x_counterfactual_transformed = self.x_counterfactual
    #     else:
    #         # For ECT matcher or non-pipeline models, use original data
    #         x_factual_transformed = self.x_factual
    #         x_counterfactual_transformed = self.x_counterfactual

    #     if self.matcher == "ot":
    #         return CounterfactualOptimalTransportPolicy(
    #             x_factual_transformed, x_counterfactual_transformed
    #         ).compute_prob_matrix_of_factual_and_counterfactual()
    #     elif self.matcher == "cem":
    #         return CounterfactualSoftCEMPolicy(
    #             x_factual_transformed, x_counterfactual_transformed
    #         ).compute_prob_matrix_of_factual_and_counterfactual()
    #     elif self.matcher == "nn":
    #         return CounterfactualNearestNeighborMatchingPolicy(
    #             x_factual_transformed, x_counterfactual_transformed
    #         ).compute_prob_matrix_of_factual_and_counterfactual()
    #     elif self.matcher == "ect":
    #         return CounterfactualExactMatchingPolicy(
    #             x_factual_transformed, x_counterfactual_transformed
    #         ).compute_prob_matrix_of_factual_and_counterfactual()
    

    def _get_matcher(self):
        """
        Get matcher based on policy.

        如果 COLAData 提供了转换后的数据（has_transformed_data=True），
        则使用转换后的数据进行 matching 计算，以确保在转换空间中计算距离。
        否则，使用预处理方法转换数据（例如从 pipeline 中提取 preprocessor）。
        """

        # 优先使用 COLAData 提供的转换后数据
        if self.has_transformed_data:
            x_factual_for_matching = self.x_factual_transformed
            x_counterfactual_for_matching = self.x_counterfactual_transformed
        else:
            # 如果没有转换后的数据，使用原有的预处理方法
            # （例如从 pipeline 中提取 preprocessor 进行转换）
            x_factual_for_matching = self._preprocess_for_matching(self.x_factual_pandas)
            x_counterfactual_for_matching = self._preprocess_for_matching(self.x_counterfactual_pandas)

        if self.matcher == "ot":
            return CounterfactualOptimalTransportPolicy(
                x_factual_for_matching, x_counterfactual_for_matching
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "cem":
            return CounterfactualSoftCEMPolicy(
                x_factual_for_matching, x_counterfactual_for_matching
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "nn":
            return CounterfactualNearestNeighborMatchingPolicy(
                x_factual_for_matching, x_counterfactual_for_matching
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "ect":
            return CounterfactualExactMatchingPolicy(
                x_factual_for_matching, x_counterfactual_for_matching
            ).compute_prob_matrix_of_factual_and_counterfactual()

    def _preprocess_for_matching(self, data):
        """
        Preprocess data for matching by transforming to numerical format.
        
        Args:
            data: Input data (pandas DataFrame or numpy array)
            
        Returns:
            Numerical numpy array suitable for distance calculations
        """
        
        # If already a numerical numpy array, return as is
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            return data
        
        # If DataFrame with all numerical columns, convert to array
        if isinstance(data, pd.DataFrame):
            if data.select_dtypes(include=['object', 'string']).shape[1] == 0:
                return data.values
        
        # Try to use ml_model's preprocessor for transformation
        if hasattr(self.ml_model, 'pipeline'):
            try:
                # Check if preprocessor exists in pipeline
                if hasattr(self.ml_model.pipeline, 'named_steps') and 'preprocessor' in self.ml_model.pipeline.named_steps:
                    preprocessor = self.ml_model.pipeline.named_steps['preprocessor']
                    transformed = preprocessor.transform(data)
                    
                    # Convert sparse matrix to dense if needed
                    if hasattr(transformed, 'toarray'):
                        transformed = transformed.toarray()
                    
                    return transformed
                
                # Alternative: check using dict
                elif 'preprocessor' in dict(self.ml_model.pipeline.steps):
                    preprocessor = dict(self.ml_model.pipeline.steps)['preprocessor']
                    transformed = preprocessor.transform(data)
                    
                    # Convert sparse matrix to dense if needed
                    if hasattr(transformed, 'toarray'):
                        transformed = transformed.toarray()
                    
                    return transformed
                    
            except Exception as e:
                print(f"Warning: Failed to use preprocessor for transformation: {e}")
                print("Attempting fallback conversion...")
        
        # Fallback: try to convert DataFrame to float
        if isinstance(data, pd.DataFrame):
            try:
                # Try converting string numbers to float
                return data.apply(pd.to_numeric, errors='coerce').values
            except Exception as e:
                print(f"Warning: Failed to convert DataFrame to numerical: {e}")
        
        # Last resort: return as numpy array
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _get_attributor(self):
        """
        Get attribution based on policy.

        如果 COLAData 提供了转换后的数据，则使用转换后的数据计算 varphi。
        """
        if self.attributor == "pshap":
            # 选择使用原始数据还是转换后的数据
            if self.has_transformed_data:
                x_factual_for_varphi = self.x_factual_transformed
                x_counterfactual_for_varphi = self.x_counterfactual_transformed
                # 使用转换后的列名
                feature_names = self.data.get_transformed_feature_columns()
            else:
                x_factual_for_varphi = self.x_factual
                x_counterfactual_for_varphi = self.x_counterfactual
                # 使用原始列名
                feature_names = self.data.get_feature_columns()

            varphi = PSHAP(
                ml_model=self.ml_model,
                x_factual=x_factual_for_varphi,
                x_counterfactual=x_counterfactual_for_varphi,
                joint_prob=self._get_matcher(),
                random_state=self.random_state,
                feature_names=feature_names
            ).calculate_varphi()
            return varphi
    
    def _get_data_composer(self):
        """
        Get data composer based on policy.

        如果 COLAData 提供了转换后的数据，则使用转换后的数据计算 q。
        """
        # 选择使用原始数据还是转换后的数据
        if self.has_transformed_data:
            x_counterfactual_for_q = self.x_counterfactual_transformed
        else:
            x_counterfactual_for_q = self.x_counterfactual

        q = DataComposer(
            x_counterfactual=x_counterfactual_for_q,
            joint_prob=self._get_matcher(),
        ).calculate_q()

        return q
    
    def _to_dataframe(self, x):
        """
        Convert numpy array to DataFrame with feature names.

        Parameters:
        -----------
        x : np.ndarray
            Input data as numpy array

        Returns:
        --------
        pd.DataFrame
            DataFrame with proper column names
        """
        # Always return a DataFrame copy so callers can safely modify it
        if isinstance(x, pd.DataFrame):
            df = x.copy()
        else:
            df = pd.DataFrame(x, columns=self.data.get_feature_columns())

        # If numerical features are defined in COLAData, treat the remaining
        # feature columns as categorical and cast them to string. This keeps
        # a consistent dtype expectation between training and runtime (e.g. for
        # OneHotEncoder / DiCE) and avoids np.isnan/type issues inside sklearn.
        try:
            num_feats = self.data.get_numerical_features() if hasattr(self.data, 'get_numerical_features') else []
            label_col = self.data.get_label_column() if hasattr(self.data, 'get_label_column') else None
            cat_cols = [c for c in df.columns if c != label_col and c not in (num_feats or [])]
            for c in cat_cols:
                # cast to str in-place; ignore if conversion fails
                try:
                    df[c] = df[c].astype(str)
                except Exception:
                    pass
        except Exception:
            # Be robust: if anything goes wrong, just return the DataFrame as-is
            return df

        return df

    def _return_dataframe(self, x, y, index=None):
        """
        Convert numpy array to DataFrame with labels.

        Parameters:
        -----------
        x : np.ndarray
            Feature data
        y : np.ndarray
            Target predictions
        index : pd.Index, optional
            Index for the DataFrame. If the length doesn't match x,
            a new RangeIndex will be created instead.

        Returns:
        --------
        pd.DataFrame
            DataFrame with features and target column
        """
        # Check if index length matches data length
        if index is not None and len(index) != len(x):
            # Index length mismatch, create new index
            index = None

        df = pd.DataFrame(x, index=index)
        df.columns = self.data.get_feature_columns()
        df[self.data.get_label_column()] = y
        return df
    
    def _get_action_sequence(self, varphi, m, allowed_col_indices: Optional[np.ndarray] = None):
        """
        Get action sequence for sampling counterfactual modifications.

        Parameters:
        -----------
        varphi : np.ndarray
            Attribution matrix (probability distribution over actions)
        m : int
            Number of actions to sample
        allowed_col_indices : np.ndarray, optional
            Array of column indices that are allowed to be modified.
            If None, all columns can be modified (default).
            If specified, only actions involving these columns will be considered.

        Returns:
        --------
        tuple
            (row_indices, col_indices) - Arrays of row and column indices for actions
        """
        # Ensure varphi is 2D
        if varphi.ndim == 1:
            varphi = varphi.reshape(1, -1)
        elif varphi.ndim > 2:
            # If varphi has more than 2 dimensions, flatten to 2D
            # Assuming first dimension is samples, rest should be features
            varphi = varphi.reshape(varphi.shape[0], -1)

        rng = np.random.RandomState(self.random_state)
        
        if allowed_col_indices is not None:
            # Create a mask for allowed actions (only actions in allowed columns)
            varphi_flat = varphi.flatten()
            varphi_reshaped = varphi.reshape(varphi.shape[0], varphi.shape[1])
            
            # Create a mask: True for allowed columns, False otherwise
            mask = np.zeros(varphi.shape, dtype=bool)
            mask[:, allowed_col_indices] = True
            
            # Flatten the mask
            mask_flat = mask.flatten()
            
            # Set probability to 0 for disallowed actions
            varphi_filtered = varphi_flat.copy()
            varphi_filtered[~mask_flat] = 0
            
            # Normalize probabilities
            if varphi_filtered.sum() == 0:
                raise ValueError(
                    "No valid actions found with the specified features_to_vary. "
                    "Please check that the feature names are correct and that there are valid actions."
                )
            varphi_filtered = varphi_filtered / varphi_filtered.sum()
            
            # Sample from filtered distribution
            # Get all valid action indices
            valid_action_indices = np.where(mask_flat)[0]
            if len(valid_action_indices) < m:
                # If there are fewer valid actions than requested, use all valid actions
                actions_sequence = valid_action_indices
            else:
                # Sample m actions from valid actions
                valid_probs = varphi_filtered[valid_action_indices]
                valid_probs = valid_probs / valid_probs.sum()  # Normalize
                actions_sequence = rng.choice(
                    a=valid_action_indices, size=m, p=valid_probs, replace=False
                )
            
            row_indices, col_indices = np.unravel_index(actions_sequence, varphi.shape)
        else:
            # Original behavior: sample from all actions
            actions_sequence = rng.choice(
                a=varphi.size, size=m, p=varphi.flatten(), replace=False
            )
            row_indices, col_indices = np.unravel_index(actions_sequence, varphi.shape)
        
        return row_indices, col_indices
    
    def _compute_minimum_actions(self, varphi, q, y_corresponding_counterfactual, allowed_col_indices: Optional[np.ndarray] = None):
        """
        Compute the minimum number of actions needed to achieve target predictions.
        
        Parameters:
        -----------
        varphi : np.ndarray
            Attribution matrix
        q : np.ndarray
            Data composer result
        y_corresponding_counterfactual : np.ndarray
            Target predictions
        allowed_col_indices : np.ndarray, optional
            Allowed column indices if features_to_vary is specified
        
        Returns:
        --------
        tuple
            (minimum_actions, is_feasible)
            - minimum_actions: int - Minimum number of actions required (or maximum available if not feasible)
            - is_feasible: bool - True if target can be achieved, False if even using all actions cannot achieve target
        """
        # Calculate maximum possible actions
        # If allowed_col_indices is specified, max actions = number of different positions
        # between factual and counterfactual in allowed features only
        # Otherwise, max actions = all possible actions (varphi.size)
        # 注意：如果使用了转换，则使用转换后的 factual 数据
        if self.has_transformed_data:
            x_factual_for_calc = self.x_factual_transformed
        else:
            x_factual_for_calc = self.x_factual

        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = x_factual_for_calc[:, allowed_col_indices]
            counterfactual_allowed = q[:, allowed_col_indices]
            # Count all positions where values differ
            diff_mask = factual_allowed != counterfactual_allowed
            m = np.sum(diff_mask)
        else:
            # m 定义为 actions 最大可能个数（所有特征 × 所有样本）
            m = varphi.size
        
        # If m is 0, no actions are possible
        if m == 0:
            return 0, False
        
        low, high = 1, m
        mid = m  # Initialize with max in case binary search doesn't find a solution
        is_feasible = True
        
        while low < high:
            mid = (low + high) // 2
            diff = self._compare_target_ce_to_ace(mid, y_corresponding_counterfactual, varphi, q, allowed_col_indices)
            if diff != 0:
                # Not enough actions, need more - increase mid until diff becomes 0
                while diff != 0 and mid < m:
                    mid = mid + 1
                    diff = self._compare_target_ce_to_ace(mid, y_corresponding_counterfactual, varphi, q, allowed_col_indices)
                
                # Check if we reached maximum but still can't achieve target
                if mid >= m and diff != 0:
                    # Cannot achieve target even with all available actions
                    is_feasible = False
                
                # After finding the minimum (or reaching max), break
                break
            else:
                # Found a valid solution (diff == 0), try to find smaller one
                high = mid
        
        return mid, is_feasible
    
    def _compare_target_ce_to_ace(self, mid, y_corresponding_counterfactual, varphi, q, allowed_col_indices: Optional[np.ndarray] = None):
        """
        Compare target counterfactual predictions with action-limited counterfactual predictions.
        
        Parameters:
        -----------
        mid : int
            Number of actions to test
        y_corresponding_counterfactual : np.ndarray
            Target predictions
        varphi : np.ndarray
            Attribution matrix
        q : np.ndarray
            Data composer result
        allowed_col_indices : np.ndarray, optional
            Allowed column indices if features_to_vary is specified
        
        Returns:
        --------
        int
            Number of samples where predictions don't match (0 means all match)
        """
        # Get maximum possible actions (considering allowed_col_indices if specified)
        # If allowed_col_indices is specified, count actual different positions between factual and counterfactual
        # 注意：如果使用了转换，则使用转换后的 factual 数据
        if self.has_transformed_data:
            x_factual_for_calc = self.x_factual_transformed
        else:
            x_factual_for_calc = self.x_factual

        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = x_factual_for_calc[:, allowed_col_indices]
            counterfactual_allowed = q[:, allowed_col_indices]
            diff_mask = factual_allowed != counterfactual_allowed
            total_actions = np.sum(diff_mask)
        else:
            total_actions = varphi.size

        all_row_indices, all_col_indices = self._get_action_sequence(varphi, total_actions, allowed_col_indices)
        row_indices = all_row_indices[:mid]
        col_indices = all_col_indices[:mid]
        q_values = q[row_indices, col_indices]
        x_action_constrained = x_factual_for_calc.copy()
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val

        # 使用转换空间的数据进行预测
        if self.has_transformed_data:
            y_counterfactual_limited_actions = self.ml_model.predict(
                pd.DataFrame(x_action_constrained, columns=self.data.get_transformed_feature_columns())
            )
        else:
            y_counterfactual_limited_actions = self.ml_model.predict(self._to_dataframe(x_action_constrained))

        result = np.sum(y_counterfactual_limited_actions != y_corresponding_counterfactual)
        return result
    
    def _find_changes_m_of_ce(self):
        """Find the number of changes in CE."""
        corresponding_counterfactual = self._get_data_composer()
        y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(corresponding_counterfactual))
        m = np.sum(corresponding_counterfactual != self.x_factual)
        return m, y_corresponding_counterfactual

