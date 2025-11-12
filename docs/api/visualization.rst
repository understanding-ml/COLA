==================
Visualization API
==================

.. currentmodule:: xai_cola.ce_sparsifier.visualization

Overview
========

COLA provides a suite of pure functions for visualizing counterfactual refinement results.
These functions are stateless and can be used independently, but **it is recommended to call
them through COLA class methods**, as the COLA class automatically manages the required data.

**Recommended Usage:**

Call through COLA class methods (see :doc:`cola` documentation):

- ``cola.heatmap_binary()`` - Binary heatmap
- ``cola.heatmap_direction()`` - Direction heatmap
- ``cola.stacked_bar_chart()`` - Stacked bar chart
- ``cola.highlight_changes_comparison()`` - Comparison highlight table
- ``cola.highlight_changes_final()`` - Final value highlight table
- ``cola.diversity()`` - Diversity analysis

**Direct Usage:**

Advanced users can import functions directly from this module for greater flexibility.
This documentation describes the signatures and parameters of these low-level functions.

**Available Visualization Types:**

1. **Highlighted DataFrames** - Interactive tables with color-coded changes
2. **Direction Heatmaps** - Show which features increased/decreased
3. **Binary Heatmaps** - Show which features changed (yes/no)
4. **Stacked Bar Charts** - Compare feature change counts before/after refinement
5. **Diversity Analysis** - Explore alternative minimal feature combinations

Module contents
===============

Highlighted DataFrames
======================

function **highlight_changes_comparison** (factual_df, counterfactual_df, refined_counterfactual_df, label_column)

   Pure function to highlight changes with comparison format (old -> new).

   This function displays changes in the format "factual_value -> counterfactual_value"
   to show both the original and modified values side by side.

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data
      * **counterfactual_df** (*pd.DataFrame*) -- Full counterfactual data (corresponding counterfactual)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Action-limited counterfactual data
      * **label_column** (*str*) -- Name of the target/label column

   **Returns:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data (copy)
      * **ce_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing factual → full counterfactual
      * **ace_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing factual → action-limited counterfactual

   **Return type:**
      tuple

   **Color Coding:**

   - Yellow background: Feature value changed (except label column)
   - Light gray background + black border: Label column changed
   - No highlight: Value unchanged

   **Note:**

   Typically called through ``cola.highlight_changes_comparison()`` method, which automatically provides the required data.

   **Example:**

   .. code-block:: python

      from xai_cola.ce_sparsifier.visualization import highlight_changes_comparison

      factual_copy, ce_style, ace_style = highlight_changes_comparison(
          factual_df=factual,
          counterfactual_df=counterfactual,
          refined_counterfactual_df=refined_cf,
          label_column='Risk'
      )

      # Display in Jupyter
      display(ce_style)
      display(ace_style)

function **highlight_changes_final** (factual_df, counterfactual_df, refined_counterfactual_df, label_column)

   Pure function to highlight changes showing only the final values.

   This function displays only the final counterfactual values without showing
   the "factual -> counterfactual" format, making it cleaner for presentation.

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data
      * **counterfactual_df** (*pd.DataFrame*) -- Full counterfactual data (corresponding counterfactual)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Action-limited counterfactual data
      * **label_column** (*str*) -- Name of the target/label column

   **Returns:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data (copy)
      * **ce_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing only full counterfactual values
      * **ace_style** (*pandas.io.formats.style.Styler*) -- Styled DataFrame showing only action-limited counterfactual values

   **Return type:**
      tuple

   **Color Coding:**

   Same as ``highlight_changes_comparison()``:

   - Yellow background: Feature value changed
   - Light gray background + black border: Label column changed
   - No highlight: Value unchanged

   **Note:**

   Typically called through ``cola.highlight_changes_final()`` method.

function **highlight_differences** (data, df_a, df_b, target_name)

   Low-level pure function to highlight differences between two DataFrames.

   Creates a style DataFrame with background colors and borders to highlight
   differences between the two input DataFrames. Used internally by other
   highlight functions.

   **Parameters:**
      * **data** (*pd.DataFrame*) -- The DataFrame to style (should match df_a and df_b structure)
      * **df_a** (*pd.DataFrame*) -- First DataFrame for comparison (typically factual)
      * **df_b** (*pd.DataFrame*) -- Second DataFrame for comparison (typically counterfactual)
      * **target_name** (*str*) -- Name of the target/label column

   **Returns:**
      Style DataFrame with CSS styling strings for highlighting differences

   **Return type:**
      pd.DataFrame

Binary Heatmap
==============

function **generate_binary_heatmap** (factual_df, counterfactual_df, refined_counterfactual_df, label_column, save_path=None, save_mode='combined', show_axis_labels=True)

   Pure function to generate binary change heatmap visualizations.

   This function generates heatmaps that show binary changes: whether a value
   changed (red) or remained unchanged (lightgrey).

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data
      * **counterfactual_df** (*pd.DataFrame*) -- Full counterfactual data (corresponding counterfactual)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Action-limited counterfactual data
      * **label_column** (*str*) -- Name of the target/label column
      * **save_path** (*str**, **optional*) -- Path to save the heatmap images.
        If None, plots are automatically displayed in Jupyter (default).
        If provided, saves to the specified path and closes the plots.
        Can be a directory path or file path.
      * **save_mode** (*str**, **default='combined'*) -- How to save the heatmaps when save_path is provided:
        - "combined": Save both heatmaps in a single combined image (top and bottom)
        - "separate": Save two separate image files (heatmap_ce.png and heatmap_ace.png)
        - Ignored if save_path is None
      * **show_axis_labels** (*bool**, **default=True*) -- Whether to show x and y axis labels (column names and row indices).
        If True, displays column names and row indices. If False, hides them.

   **Returns:**
      (plot1, plot2) - Heatmap plots (matplotlib Figure objects)

   **Return type:**
      tuple

   **Color Coding:**

   - Light grey: Value unchanged
   - Red: Feature value changed
   - Dark blue (#000080): Label column changed

   **Note:**

   Typically called through ``cola.heatmap_binary()`` method.

function **heatmap_binary_changes** (factual, counterfactual, target_name, background_color='lightgrey', changed_feature_color='red', show_axis_labels=False)

   Pure function to generate binary change heatmap (shows if value changed or not).

   This function generates a heatmap that shows binary changes: whether a value
   changed (red) or remained unchanged (lightgrey). For the target column, changed
   values are shown in dark blue.

   **Parameters:**
      * **factual** (*pd.DataFrame**, **optional*) -- Factual DataFrame
      * **counterfactual** (*pd.DataFrame**, **optional*) -- Counterfactual DataFrame
      * **target_name** (*str**, **optional*) -- Name of the target/label column
      * **background_color** (*str**, **optional*) -- Background color for unchanged cells. Defaults to 'lightgrey'.
      * **changed_feature_color** (*str**, **optional*) -- Color for changed features. Defaults to 'red'.
      * **show_axis_labels** (*bool**, **optional*) -- Whether to show x and y axis labels. Defaults to False.

   **Returns:**
      The heatmap figure for changes from factual to counterfactual

   **Return type:**
      matplotlib.figure.Figure

Direction Heatmap
=================

function **generate_direction_heatmap** (factual_df, counterfactual_df, refined_counterfactual_df, label_column, numerical_features=None, save_path=None, save_mode='combined', show_axis_labels=True)

   Pure function to generate directional change heatmap visualizations with distinction between numerical and categorical features.

   This function generates heatmaps that show the direction of changes:

   - Numerical features (increased): light blue (#56B4E9)
   - Numerical features (decreased): light cyan (#7FCDCD)
   - Categorical features (changed): peru (#CD853F)
   - Value unchanged: lightgrey
   - Target column: changed values shown in black

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data
      * **counterfactual_df** (*pd.DataFrame*) -- Full counterfactual data (corresponding counterfactual)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Action-limited counterfactual data
      * **label_column** (*str*) -- Name of the target/label column
      * **numerical_features** (*list**, **optional*) -- List of numerical feature names.
        If None, all features (except label) are treated as numerical.
      * **save_path** (*str**, **optional*) -- Path to save the heatmap images.
        If None, plots are automatically displayed in Jupyter (default).
        If provided, saves to the specified path and closes the plots.
        Can be a directory path or file path.
      * **save_mode** (*str**, **default='combined'*) -- How to save the heatmaps when save_path is provided:
        - "combined": Save both heatmaps in a single combined image (top and bottom)
        - "separate": Save two separate image files
        - Ignored if save_path is None
      * **show_axis_labels** (*bool**, **default=True*) -- Whether to show x and y axis labels (column names and row indices).
        If True, displays column names and row indices. If False, hides them.

   **Returns:**
      (plot1, plot2) - Heatmap plots (matplotlib Figure objects)

   **Return type:**
      tuple

   **Note:**

   Typically called through ``cola.heatmap_direction()`` method.

function **heatmap_direction_changes** (factual, counterfactual, target_name, numerical_features=None, unchanged_color='lightgrey', increased_color='#56B4E9', decreased_color='#7FCDCD', categorical_changed_color='#CD853F', show_axis_labels=False)

   Pure function to generate directional change heatmap with distinction between numerical and categorical features.

   This function generates a heatmap that shows the direction of changes:

   - Numerical features (increased): light blue (#56B4E9)
   - Numerical features (decreased): light cyan (#7FCDCD)
   - Categorical features (changed): peru (#CD853F)
   - Value unchanged: lightgrey
   - Target column: changed values shown in black (unchanged remain lightgrey)

   **Parameters:**
      * **factual** (*pd.DataFrame**, **optional*) -- Factual DataFrame
      * **counterfactual** (*pd.DataFrame**, **optional*) -- Counterfactual DataFrame
      * **target_name** (*str**, **optional*) -- Name of the target/label column
      * **numerical_features** (*list**, **optional*) -- List of numerical feature names.
        If None, all features are treated as numerical.
      * **unchanged_color** (*str**, **optional*) -- Background color for unchanged cells. Defaults to 'lightgrey'.
      * **increased_color** (*str**, **optional*) -- Color for increased values (numerical only). Defaults to '#56B4E9' (light blue).
      * **decreased_color** (*str**, **optional*) -- Color for decreased values (numerical only). Defaults to '#7FCDCD' (light cyan).
      * **categorical_changed_color** (*str**, **optional*) -- Color for categorical feature changes. Defaults to '#CD853F' (peru).
      * **show_axis_labels** (*bool**, **optional*) -- Whether to show x and y axis labels. Defaults to False.

   **Returns:**
      The heatmap figure showing direction of changes from factual to counterfactual

   **Return type:**
      matplotlib.figure.Figure

Stacked Bar Chart
=================

function **generate_stacked_bar_chart** (factual_df, counterfactual_df, refined_counterfactual_df, label_column, save_path=None, refined_color='#D9F2D0', counterfactual_color='#FBE3D6', instance_labels=None)

   Pure function to generate and optionally save a stacked percentage bar chart.

   This is a wrapper around create_stacked_bar_chart() for consistency with other
   visualization functions.

   Creates a percentage-based stacked bar chart where each bar shows:

   - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
   - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
   - Total bar length: 100% (representing all counterfactual modifications)

   Labels show both percentage and actual count (e.g., "60.0% (3)")

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data
      * **counterfactual_df** (*pd.DataFrame*) -- Full counterfactual data (corresponding counterfactual)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Action-limited counterfactual data
      * **label_column** (*str*) -- Name of the target/label column to exclude from comparison
      * **save_path** (*str**, **optional*) -- Path to save the chart image. If None, chart is not saved.
      * **refined_color** (*str**, **default='#D9F2D0'*) -- Color for refined counterfactual modified positions (light green)
      * **counterfactual_color** (*str**, **default='#FBE3D6'*) -- Color for counterfactual modified positions (light pink/orange)
      * **instance_labels** (*list**, **optional*) -- Custom labels for instances

   **Returns:**
      The stacked bar chart figure

   **Return type:**
      matplotlib.figure.Figure

   **Note:**

   Typically called through ``cola.stacked_bar_chart()`` method.

function **create_stacked_bar_chart** (factual_df, counterfactual_df, refined_counterfactual_df, label_column, save_path=None, refined_color='#D9F2D0', counterfactual_color='#FBE3D6', instance_labels=None)

   Pure function to create a horizontal stacked percentage bar chart comparing modification positions.

   This function creates a percentage-based stacked bar chart where each bar represents
   an instance (100% total), showing the proportion of modified positions in refined
   counterfactual vs. original counterfactual relative to factual data.

   Each bar shows:

   - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
   - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
   - Total bar length: 100% (representing all counterfactual modifications)

   Labels on bars show both percentage and actual count (e.g., "60.0% (3)")

   **Parameters:**
      Same as ``generate_stacked_bar_chart()``

   **Returns:**
      The stacked bar chart figure

   **Return type:**
      matplotlib.figure.Figure

Diversity Analysis
==================

function **generate_diversity_for_all_instances** (factual_df, refined_counterfactual_df, ml_model, label_column, cola_data=None)

   Generate diversity analysis for all instances.

   This is the main pure function that processes all instances. For each instance,
   it finds minimal feature combinations that can flip the prediction from factual's
   target value to refined counterfactual's target value (e.g., from 1 to 0).

   **Parameters:**
      * **factual_df** (*pd.DataFrame*) -- Factual data (with label column)
      * **refined_counterfactual_df** (*pd.DataFrame*) -- Refined counterfactual data (with label column)
      * **ml_model** (*Model*) -- ML model for prediction
      * **label_column** (*str*) -- Name of target column
      * **cola_data** (*COLAData**, **optional*) -- COLAData object for data transformation (needed when using transform_method)

   **Returns:**
      * **factual_df** (*pd.DataFrame*) -- Original factual data (copy)
      * **diversity_styles** (*List[Styler]*) -- List of styled DataFrames (one per instance),
        each showing all minimal feature combinations for that instance

   **Return type:**
      Tuple[pd.DataFrame, List[Styler]]

   **Note:**

   Typically called through ``cola.diversity()`` method.

function **find_minimal_feature_combinations** (factual_row, refined_counterfactual_row, ml_model, label_column, cola_data=None)

   Find all minimal feature combinations that can flip the target from factual to refined counterfactual.

   This function finds minimal sets of features that, when changed from factual to refined counterfactual
   values, will cause the model prediction to change from factual's target value to refined counterfactual's
   target value (e.g., from 1 to 0).

   **Parameters:**
      * **factual_row** (*pd.Series*) -- Single factual instance (with label column)
      * **refined_counterfactual_row** (*pd.Series*) -- Single refined counterfactual instance (with label column)
      * **ml_model** (*Model*) -- ML model for prediction
      * **label_column** (*str*) -- Name of target column
      * **cola_data** (*COLAData**, **optional*) -- COLAData object for data transformation (needed when using transform_method)

   **Returns:**
      List of minimal feature combinations (each is a set of feature names)

   **Return type:**
      List[Set[str]]

   **Algorithm:**

   Uses exhaustive enumeration starting from 1 feature up to all changed features.
   Ensures true minimality: if changing feature A alone works, combinations
   like {A, B} are excluded.

function **generate_diversity_dataframe** (factual_row, refined_counterfactual_row, minimal_combinations, label_column)

   Generate diversity DataFrame with all minimal combinations.

   The first row shows the refined counterfactual (all changes from refined CF), followed by
   rows showing each minimal combination. If a minimal combination modifies exactly the same
   features as the refined counterfactual, it is excluded since it provides no additional diversity.

   **Parameters:**
      * **factual_row** (*pd.Series*) -- Single factual instance
      * **refined_counterfactual_row** (*pd.Series*) -- Single refined counterfactual instance
      * **minimal_combinations** (*List[Set[str]]*) -- List of minimal feature combinations
      * **label_column** (*str*) -- Name of target column

   **Returns:**
      DataFrame with first row as refined counterfactual,
      followed by one row per minimal combination (excluding duplicates)

   **Return type:**
      pd.DataFrame

function **highlight_diversity_changes** (factual_row, diversity_df, label_column)

   Highlight changes in diversity DataFrame.

   The first row shows the refined counterfactual with all changes highlighted,
   with yellow background to distinguish it from minimal combinations.
   Following rows show minimal combinations with only their specific changes highlighted.

   **Parameters:**
      * **factual_row** (*pd.Series*) -- Single factual instance
      * **diversity_df** (*pd.DataFrame*) -- DataFrame with diversity combinations (first row is complete counterfactual)
      * **label_column** (*str*) -- Name of target column

   **Returns:**
      Styled DataFrame with highlighted changes

   **Return type:**
      Styler

   **Color Coding:**

   - First row (complete counterfactual): Yellow background with black border
   - Other rows (minimal combinations): #FFFFCC background with black border
   - Label column changes: Light gray background with black border
   - Unchanged values: No highlight

Examples
========

**Note:** The following examples show direct function calls. In practice, it is recommended
to call through COLA class methods (see :doc:`cola` documentation).

Highlighted DataFrames
----------------------

Comparison Format (old -> new)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import highlight_changes_comparison

   factual_copy, ce_style, ace_style = highlight_changes_comparison(
       factual_df=factual,
       counterfactual_df=counterfactual,
       refined_counterfactual_df=refined_cf,
       label_column='Risk'
   )

   # Display in Jupyter
   display(ce_style)
   display(ace_style)

   # Save to HTML
   ce_style.to_html('original_cf.html')
   ace_style.to_html('refined_ace.html')

Final Values Only
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import highlight_changes_final

   factual_copy, ce_style, ace_style = highlight_changes_final(
       factual_df=factual,
       counterfactual_df=counterfactual,
       refined_counterfactual_df=refined_cf,
       label_column='Risk'
   )

   display(ace_style)

Binary Heatmap
--------------

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import generate_binary_heatmap

   plot1, plot2 = generate_binary_heatmap(
       factual_df=factual,
       counterfactual_df=counterfactual,
       refined_counterfactual_df=refined_cf,
       label_column='Risk',
       save_path='./results',
       save_mode='combined',
       show_axis_labels=True
   )

Direction Heatmap
-----------------

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import generate_direction_heatmap

   plot1, plot2 = generate_direction_heatmap(
       factual_df=factual,
       counterfactual_df=counterfactual,
       refined_counterfactual_df=refined_cf,
       label_column='Risk',
       numerical_features=['Age', 'Income', 'Credit amount'],
       save_path='./results',
       save_mode='combined',
       show_axis_labels=True
   )

Stacked Bar Chart
-----------------

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import generate_stacked_bar_chart

   fig = generate_stacked_bar_chart(
       factual_df=factual,
       counterfactual_df=counterfactual,
       refined_counterfactual_df=refined_cf,
       label_column='Risk',
       save_path='./results',
       instance_labels=['Sample 1', 'Sample 2', 'Sample 3']
   )

Diversity Analysis
------------------

.. code-block:: python

   from xai_cola.ce_sparsifier.visualization import generate_diversity_for_all_instances

   factual_copy, diversity_styles = generate_diversity_for_all_instances(
       factual_df=factual,
       refined_counterfactual_df=refined_cf,
       ml_model=ml_model,
       label_column='Risk',
       cola_data=data  # Optional, needed if using transform_method
   )

   # Display results for each instance
   for i, style in enumerate(diversity_styles):
       print(f"Instance {i+1} diversity:")
       display(style)

See Also
========

- :doc:`cola` - COLA main class (recommended way to call visualizations)
- :doc:`data` - Data interface
- :doc:`models` - Model interface
