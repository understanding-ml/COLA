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
    
    Parameters:
    -----------
    data : COLAData
        Data wrapper containing factual and counterfactual data.
        Must have both factual and counterfactual data set using add_counterfactuals().
    ml_model : Model
        Machine learning model interface
    
    Raises:
    -------
    ValueError
        If data does not have counterfactual data set
    
    Example:
    --------
    >>> from xai_cola import COLA
    >>> from xai_cola.ce_sparsifier.data import COLAData
    >>> from xai_cola.ce_sparsifier.models import Model
    >>> 
    >>> # Initialize data with factual and counterfactual
    >>> data = COLAData(factual_data=df, label_column='Risk')
    >>> data.add_counterfactuals(cf_df)  # Must add counterfactuals first
    >>> 
    >>> # Initialize model
    >>> model = Model(ml_model, backend='sklearn')
    >>> 
    >>> # Use COLA (counterfactual data is required)
    >>> cola = COLA(data=data, ml_model=model)
    >>> cola.set_policy(matcher='ect', attributor='pshap')
    >>> 
    >>> # Get only refined counterfactual
    >>> refined_cf = cola.get_refined_counterfactual(limited_actions=10)
    >>> 
    >>> # Or get all results
    >>> factual_df, counterfactual_df, refined_cf_df = cola.get_all_results(limited_actions=10)
    >>> 
    >>> # Restrict modifications to specific features only
    >>> refined_cf = cola.get_refined_counterfactual(
    ...     limited_actions=10,
    ...     features_to_vary=['Age', 'Credit amount']  # Only modify these features
    ... )
    """
    
    def __init__(
        self,
        data: COLAData,
        ml_model: Model,
        random_state: int = 42,
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
        random_state : int, optional
            Random seed for reproducible results. Default is 42.
        
        Raises:
        -------
        ValueError
            If data does not have counterfactual data set
        """
        self.data = data
        self.ml_model = ml_model
        self.random_state = random_state
        
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
        
        # Extract factual and counterfactual from data
        # Store as DataFrames to preserve column names for sklearn Pipeline compatibility
        self.x_factual_pandas = data.get_factual_features()
        self.x_counterfactual_pandas = data.get_counterfactual_features()

        # Also store numpy versions for backward compatibility
        self.x_factual = data.to_numpy_factual_features()
        self.x_counterfactual = data.to_numpy_counterfactual_features()
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
        Avalues_method: str = "max",
        random_state: Optional[int] = None,
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
        Avalues_method : str
            Method for computing A-values
            Options: "max" (maximum value method)
        random_state : int, optional
            Random seed used to control the reproducibility of counterfactual actions sampling.
            The same random_state means the same action sequence is sampled whenever counterfactuals or minimum-actions are queried.
            If None, uses the value provided at class initialization. Default is None.
        **kwargs
            Additional parameters for matcher and attributor
        """
        self.matcher = matcher
        self.attributor = attributor
        self.Avalues_method = Avalues_method
        if random_state is not None:
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
        print(f"Policy set: {attributor} with {matcher_name}, Avalues_method: {Avalues_method}")
    
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
        q = self._get_data_composer()
        
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
        y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(q))
        minimum_actions, is_feasible = self._compute_minimum_actions(varphi, q, y_corresponding_counterfactual, allowed_col_indices)
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
        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = self.x_factual[:, allowed_col_indices]
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
        
        # Apply selected actions
        q_values = q[row_indices, col_indices]
        x_action_constrained = self.x_factual.copy()
        
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val
        
        corresponding_counterfactual = q
        print(f'corresponding_counterfactual: {corresponding_counterfactual}')
        # Get predictions
        y_counterfactual = self.ml_model.predict(self._to_dataframe(self.x_counterfactual))
        y_counterfactual_limited = self.ml_model.predict(self._to_dataframe(x_action_constrained))
        y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(corresponding_counterfactual))
        
        # Convert to DataFrames
        # Use original labels from COLAData for factual_df instead of model predictions
        factual_df = self.data.get_factual_all()
        counterfactual_df = self._return_dataframe(self.x_counterfactual, y_counterfactual, factual_df.index)

        refined_counterfactual_df = self._return_dataframe(x_action_constrained, y_counterfactual_limited, factual_df.index)
        
        # Store for backward compatibility with other methods (highlight_changes_comparison, highlight_changes_final, heatmap, etc.)
        self.factual_dataframe = factual_df
        self.ce_dataframe = counterfactual_df
        self.ace_dataframe = refined_counterfactual_df
        self.corresponding_counterfactual_dataframe = self._return_dataframe(
            corresponding_counterfactual, y_corresponding_counterfactual, factual_df.index
        )
        print(f'corresponding_counterfactual_dataframe: {self.corresponding_counterfactual_dataframe}')
        
        # Apply same data types
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
        print(self.corresponding_counterfactual_dataframe)
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
        - Value increased: light green
        - Value decreased: light orange
        - Value unchanged: lightgrey
        - Target column: changed values shown in dark blue
        
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
        
        # Call pure visualization function
        return generate_direction_heatmap(
            factual_df=self.factual_dataframe,
            counterfactual_df=self.corresponding_counterfactual_dataframe,
            refined_counterfactual_df=self.ace_dataframe,
            label_column=self.data.get_label_column(),
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
            label_column=self.data.get_label_column()
        )

    # ========== Private Methods ==========
    
    def _get_matcher(self):
        """Get matcher based on policy."""
        if self.matcher == "ot":
            return CounterfactualOptimalTransportPolicy(
                self.x_factual, self.x_counterfactual
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "cem":
            return CounterfactualSoftCEMPolicy(
                self.x_factual, self.x_counterfactual
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "nn":
            return CounterfactualNearestNeighborMatchingPolicy(
                self.x_factual, self.x_counterfactual
            ).compute_prob_matrix_of_factual_and_counterfactual()
        elif self.matcher == "ect":
            return CounterfactualExactMatchingPolicy(
                self.x_factual, self.x_counterfactual
            ).compute_prob_matrix_of_factual_and_counterfactual()
    
    def _get_attributor(self):
        """Get attribution based on policy."""
        if self.attributor == "pshap":
            # Get feature names from data
            feature_names = self.data.get_feature_columns()
            varphi = PSHAP(
                ml_model=self.ml_model,
                x_factual=self.x_factual,
                x_counterfactual=self.x_counterfactual,
                joint_prob=self._get_matcher(),
                random_state=self.random_state,
                feature_names=feature_names
            ).calculate_varphi()
            return varphi
    
    def _get_data_composer(self):
        """Get data composer based on policy."""
        q = DataComposer(
            x_counterfactual=self.x_counterfactual,
            joint_prob=self._get_matcher(),
            method=self.Avalues_method
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
        """Convert numpy array to DataFrame with labels."""
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
        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = self.x_factual[:, allowed_col_indices]
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
        if allowed_col_indices is not None:
            # Count positions where factual and counterfactual differ in allowed columns only
            factual_allowed = self.x_factual[:, allowed_col_indices]
            counterfactual_allowed = q[:, allowed_col_indices]
            diff_mask = factual_allowed != counterfactual_allowed
            total_actions = np.sum(diff_mask)
        else:
            total_actions = varphi.size
            
        all_row_indices, all_col_indices = self._get_action_sequence(varphi, total_actions, allowed_col_indices)
        row_indices = all_row_indices[:mid]
        col_indices = all_col_indices[:mid]
        q_values = q[row_indices, col_indices]
        x_action_constrained = self.x_factual.copy()
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val
        y_counterfactual_limited_actions = self.ml_model.predict(self._to_dataframe(x_action_constrained))
        result = np.sum(y_counterfactual_limited_actions != y_corresponding_counterfactual)
        return result
    
    def _find_changes_m_of_ce(self):
        """Find the number of changes in CE."""
        corresponding_counterfactual = self._get_data_composer()
        y_corresponding_counterfactual = self.ml_model.predict(self._to_dataframe(corresponding_counterfactual))
        m = np.sum(corresponding_counterfactual != self.x_factual)
        return m, y_corresponding_counterfactual

