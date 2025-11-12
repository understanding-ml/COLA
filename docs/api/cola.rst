===============
COLA API
===============

.. currentmodule:: xai_cola.ce_sparsifier

Module contents
===============

class **COLA** (data, ml_model)

   **Bases:** ``object``

   COLA (COunterfactual with Limited Actions) - Main class for refining counterfactuals

   This class orchestrates the entire workflow of refining counterfactual explanations
   by limiting the number of feature changes.

   **Parameters:**
      * **data** (*COLAData*) -- Data wrapper containing factual and counterfactual data.
        Must have both factual and counterfactual data set using add_counterfactuals().
      * **ml_model** (*Model*) -- Machine learning model interface

   **Raises:**
      **ValueError** -- If data does not have counterfactual data set

   **Example:**

   .. code-block:: python

      from xai_cola import COLA
      from xai_cola.ce_sparsifier.data import COLAData
      from xai_cola.ce_sparsifier.models import Model

      # Initialize data with factual and counterfactual
      data = COLAData(factual_data=df, label_column='Risk')
      data.add_counterfactuals(cf_df)  # Must add counterfactuals first

      # Initialize model
      model = Model(ml_model, backend='sklearn')

      # Use COLA (counterfactual data is required)
      cola = COLA(data=data, ml_model=model)

      # Set policy with random_state for reproducibility
      cola.set_policy(matcher='ect', attributor='pshap', random_state=42)

      # Get only refined counterfactual
      refined_cf = cola.get_refined_counterfactual(limited_actions=10)

      # Or get all results
      factual_df, counterfactual_df, refined_cf_df = cola.get_all_results(limited_actions=10)

      # Restrict modifications to specific features only
      refined_cf = cola.get_refined_counterfactual(
          limited_actions=10,
          features_to_vary=['Age', 'Credit amount']  # Only modify these features
      )

   **set_policy** (matcher='ot', attributor='pshap', random_state=42, \*\*kwargs)

      Set the refinement policy.

      **Parameters:**
         * **matcher** (*str*) -- Matching strategy between factual and counterfactual.
           Options: "ot" (Optimal Transport), "ect" (Exact Matching),
           "nn" (Nearest Neighbor), "cem" (Coarsened Exact Matching). Default is "ot".
         * **attributor** (*str*) -- Feature attribution method.
           Options: "pshap" (PSHAP with joint probability). Default is "pshap".
         * **random_state** (*int**, **optional*) -- Random seed used to control the reproducibility of counterfactual actions sampling.
           The same random_state means the same action sequence is sampled whenever counterfactuals or minimum-actions are queried.
           Default is 42.
         * **\*\*kwargs** -- Additional parameters for matcher and attributor

   **get_refined_counterfactual** (limited_actions, features_to_vary=None)

      Get counterfactuals refined with limited actions.

      **Parameters:**
         * **limited_actions** (*int*) -- Maximum number of feature changes to apply
         * **features_to_vary** (*List[str]**, **optional*) -- List of feature names that are allowed to be modified.
           If None, all features can be modified (default).
           If specified, only these features will be considered for modification.

      **Returns:**
         Refined counterfactual DataFrame with target column

      **Return type:**
         pd.DataFrame

      **Raises:**
         **ValueError** -- If any feature name in features_to_vary is not a valid feature column name.

   **get_all_results** (limited_actions, features_to_vary=None)

      Get all results: factual, counterfactual, and refined counterfactual.

      **Parameters:**
         * **limited_actions** (*int*) -- Maximum number of feature changes to apply
         * **features_to_vary** (*List[str]**, **optional*) -- List of feature names that are allowed to be modified.
           If None, all features can be modified (default).
           If specified, only these features will be considered for modification.

      **Returns:**
         (factual_df, counterfactual_df, refined_counterfactual_df)
         All are pd.DataFrame with target column

      **Return type:**
         tuple

      **Raises:**
         **ValueError** -- If any feature name in features_to_vary is not a valid feature column name.

   **query_minimum_actions** (features_to_vary=None)

      Find the minimum number of actions needed.

      **Parameters:**
         **features_to_vary** (*List[str]**, **optional*) -- List of feature names that are allowed to be modified.
         If None, all features can be modified (default).
         If specified, only these features will be considered for modification.

      **Returns:**
         Minimum number of actions

      **Return type:**
         int

      **Raises:**
         **ValueError** -- If any feature name in features_to_vary is not a valid feature column name.

   **highlight_changes_comparison** ()

      Highlight changes in the dataframes with comparison format (old -> new).

      This is a thin wrapper around the pure visualization function. It retrieves
      pre-computed results from the COLA instance and passes them to the visualization
      function. Must call get_refined_counterfactual() or get_all_results() first.

      This method displays changes in the format "factual_value -> counterfactual_value"
      to show both the original and modified values side by side.

      **Returns:**
         * **factual_df** (*pandas.DataFrame*) -- Original factual data
         * **ce_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing factual → full counterfactual
         * **ace_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing factual → action-limited counterfactual

      **Return type:**
         tuple

      **Usage:**

      .. code-block:: python

         factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()
         # Display in Jupyter notebook
         display(ce_style)
         # Save as HTML
         ce_style.to_html('changes.html')

      **Raises:**
         **ValueError** -- If refined counterfactuals have not been generated yet.
         Must call get_refined_counterfactual() or get_all_results() method first.

   **highlight_changes_final** ()

      Highlight changes in the dataframes showing only the final values.

      This is a thin wrapper around the pure visualization function. It retrieves
      pre-computed results from the COLA instance and passes them to the visualization
      function. Must call get_refined_counterfactual() or get_all_results() first.

      This method displays only the final counterfactual values without showing
      the "factual -> counterfactual" format, making it cleaner for presentation.

      **Returns:**
         * **factual_df** (*pandas.DataFrame*) -- Original factual data
         * **ce_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing only full counterfactual values
         * **ace_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing only action-limited counterfactual values

      **Return type:**
         tuple

      **Usage:**

      .. code-block:: python

         factual_df, ce_style, ace_style = refiner.highlight_changes_final()
         # Display in Jupyter notebook
         display(ce_style)
         # Save as HTML
         ce_style.to_html('changes.html')

      **Raises:**
         **ValueError** -- If refined counterfactuals have not been generated yet.
         Must call get_refined_counterfactual() or get_all_results() method first.

   **heatmap_binary** (save_path=None, save_mode='combined', show_axis_labels=True)

      Generate binary change heatmap visualizations (shows if value changed or not).

      This is a thin wrapper around the pure visualization function. It retrieves
      pre-computed results from the COLA instance and passes them to the visualization
      function. Must call get_refined_counterfactual() or get_all_results() first.

      This method generates heatmaps showing binary changes: whether a value
      changed (red) or remained unchanged (lightgrey).

      **Parameters:**
         * **save_path** (*str**, **optional*) -- Path to save the heatmap images.
           If None, plots are automatically displayed in Jupyter (default).
           If provided, saves to the specified path and closes the plots.
           Can be a directory path or file path.
         * **save_mode** (*str**, **default="combined"*) -- How to save the heatmaps when save_path is provided:
           - "combined": Save both heatmaps in a single combined image (top and bottom)
           - "separate": Save two separate image files (heatmap_ce.png and heatmap_ace.png)
           - Ignored if save_path is None
         * **show_axis_labels** (*bool**, **default=True*) -- Whether to show x and y axis labels (column names and row indices).
           If True, displays column names and row indices. If False, hides them.

      **Returns:**
         (plot1, plot2) - Heatmap plots (matplotlib Figure objects)

      **Return type:**
         tuple

      **Examples:**

      .. code-block:: python

         # Display plots in Jupyter (no saving)
         refiner.heatmap_binary()
         # Plots are automatically displayed in Jupyter notebook

         # Save as combined image
         refiner.heatmap_binary(save_path='./results', save_mode='combined')
         # Creates: ./results/combined_heatmap.png (two heatmaps stacked vertically)

      **Raises:**
         **ValueError** -- If refined counterfactuals have not been generated yet.
         Must call get_refined_counterfactual() or get_all_results() method first.

   **heatmap_direction** (save_path=None, save_mode='combined', show_axis_labels=True)

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

      **Parameters:**
         * **save_path** (*str**, **optional*) -- Path to save the heatmap images.
           If None, plots are automatically displayed in Jupyter (default).
           If provided, saves to the specified path and closes the plots.
           Can be a directory path or file path.
         * **save_mode** (*str**, **default="combined"*) -- How to save the heatmaps when save_path is provided:
           - "combined": Save both heatmaps in a single combined image (top and bottom)
           - "separate": Save two separate image files
           - Ignored if save_path is None
         * **show_axis_labels** (*bool**, **default=True*) -- Whether to show x and y axis labels (column names and row indices).
           If True, displays column names and row indices. If False, hides them.

      **Returns:**
         (plot1, plot2) - Heatmap plots (matplotlib Figure objects)

      **Return type:**
         tuple

      **Examples:**

      .. code-block:: python

         # Display plots in Jupyter (no saving)
         refiner.heatmap_direction()
         # Plots are automatically displayed in Jupyter notebook

         # Save as combined image
         refiner.heatmap_direction(save_path='./results', save_mode='combined')
         # Creates: ./results/combined_direction_heatmap.png

      **Raises:**
         **ValueError** -- If refined counterfactuals have not been generated yet.
         Must call get_refined_counterfactual() or get_all_results() method first.

   **stacked_bar_chart** (save_path=None, refined_color='#D9F2D0', counterfactual_color='#FBE3D6', instance_labels=None)

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

      **Parameters:**
         * **save_path** (*str**, **optional*) -- Path to save the chart image. If None, chart is not saved.
           Can be a directory path or file path.
         * **refined_color** (*str**, **default='#D9F2D0'*) -- Color for refined counterfactual modified positions (light green)
         * **counterfactual_color** (*str**, **default='#FBE3D6'*) -- Color for counterfactual modified positions (light pink/orange)
         * **instance_labels** (*list**, **optional*) -- Custom labels for instances. If None, uses "instance 1", "instance 2", etc.
           Length must match the number of instances.

      **Returns:**
         The stacked bar chart figure

      **Return type:**
         matplotlib.figure.Figure

      **Examples:**

      .. code-block:: python

         # Display chart in Jupyter (no saving)
         fig = refiner.stacked_bar_chart()
         # Chart is automatically displayed in Jupyter notebook

         # Save chart to file
         fig = refiner.stacked_bar_chart(save_path='./results')
         # Creates: ./results/stacked_bar_chart.png

         # Custom instance labels and colors
         fig = refiner.stacked_bar_chart(
             instance_labels=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'],
             refined_color='#D9F2D0',
             counterfactual_color='#FBE3D6'
         )

      **Raises:**
         **ValueError** -- If refined counterfactuals have not been generated yet.
         Must call get_refined_counterfactual() or get_all_results() method first.

   **diversity** ()

      Generate diversity analysis showing minimal feature combinations that can flip the target.

      This method finds all minimal feature combinations that can change the prediction
      from the factual's target value to the refined counterfactual's target value (e.g., from 1 to 0).
      For each instance, it returns styled DataFrames showing the different minimal combinations.

      The algorithm:

      1. For each instance, identify features that differ between factual and refined counterfactual
      2. Test combinations of increasing size (1 feature, 2 features, etc.)
      3. Find minimal sets that flip the prediction from factual target to refined counterfactual target
      4. Skip larger combinations that contain successful smaller combinations

      **Returns:**
         * **factual_df** (*pd.DataFrame*) -- Original factual data (copy)
         * **diversity_styles** (*List[Styler]*) -- List of styled DataFrames (one per instance),
           each showing all minimal feature combinations for that instance

      **Return type:**
         Tuple[pd.DataFrame, List[Styler]]

      **Example:**

      .. code-block:: python

         factual_df, diversity_styles = sparsifier.diversity()
         # Display results for each instance
         for i, style in enumerate(diversity_styles):
             print(f"Instance {i+1} diversity:")
             display(style)
