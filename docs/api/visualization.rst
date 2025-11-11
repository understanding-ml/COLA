==================
Visualization API
==================

.. currentmodule:: xai_cola.ce_sparsifier.visualization

Overview
========

COLA provides comprehensive visualization functions for analyzing and presenting
counterfactual refinement results. All functions are stateless and can be used
independently or through the COLA class methods.

**Available Visualizations:**

1. **Highlighted DataFrames** - Interactive tables with color-coded changes
2. **Direction Heatmaps** - Show which features increased/decreased
3. **Binary Heatmaps** - Show which features changed (yes/no)
4. **Stacked Bar Charts** - Compare action counts before/after refinement
5. **Diversity Analysis** - Explore alternative minimal feature combinations

Highlighted DataFrames
======================

Functions for creating styled pandas DataFrames with highlighted changes.

.. autofunction:: highlight_changes_final

   Generate highlighted DataFrames showing only final values.

   **Color Coding:**

   - Blue background: Feature value increased
   - Orange background: Feature value decreased
   - White background: Unchanged

   **Returns:** Tuple of (factual_styled, ce_styled, ace_styled)

   **Best for:** Jupyter notebooks, HTML reports

.. autofunction:: highlight_changes_comparison

   Generate highlighted DataFrames with "old → new" format.

   Shows both original and new values for changed features.

   **Format:** ``25 → 30`` for numerical, ``No → Yes`` for categorical

   **Returns:** Tuple of (factual_styled, ce_styled, ace_styled)

   **Best for:** Detailed change analysis, presentations

.. autofunction:: highlight_differences

   Low-level helper function for highlighting differences.

   Used internally by other highlight functions.

Heatmaps
========

Direction Heatmap
-----------------

.. autofunction:: generate_direction_heatmap

   Generate heatmap showing direction of feature changes.

   **Parameters:**

   - **factual_df** (pd.DataFrame): Factual data
   - **cf_df** (pd.DataFrame): Original counterfactual data
   - **ace_df** (pd.DataFrame): Refined counterfactual data
   - **label_column** (str): Name of label column
   - **numerical_features** (List[str], optional): List of numerical features
   - **save_path** (str, optional): Directory to save figures
   - **save_mode** (str): "combined", "separate", or "both"
   - **show_axis_labels** (bool): Whether to show feature/instance names
   - **figsize** (Tuple[int, int], optional): Figure size
   - **dpi** (int, optional): Resolution

   **Color Coding:**

   - Light blue (#56B4E9): Numerical feature increased
   - Light cyan (#7FCDCD): Numerical feature decreased
   - Peru (#CD853F): Categorical feature changed
   - Black: Label column flip
   - Light grey: Unchanged

   **Returns:** Tuple[Figure, Figure] or Figure depending on save_mode

.. autofunction:: heatmap_direction_changes

   Helper function for computing direction changes.

   Used internally by generate_direction_heatmap.

Binary Heatmap
--------------

.. autofunction:: generate_binary_heatmap

   Generate binary heatmap showing which features changed.

   **Parameters:**

   - **factual_df** (pd.DataFrame): Factual data
   - **cf_df** (pd.DataFrame): Original counterfactual data
   - **ace_df** (pd.DataFrame): Refined counterfactual data
   - **label_column** (str): Name of label column
   - **save_path** (str, optional): Directory to save figures
   - **save_mode** (str): "combined", "separate", or "both"
   - **show_axis_labels** (bool): Whether to show feature/instance names
   - **figsize** (Tuple[int, int], optional): Figure size
   - **dpi** (int, optional): Resolution

   **Color Coding:**

   - Light grey: Unchanged
   - Red: Changed
   - Dark blue: Label column flip

   **Returns:** Tuple[Figure, Figure] or Figure depending on save_mode

.. autofunction:: heatmap_binary_changes

   Helper function for computing binary changes.

   Used internally by generate_binary_heatmap.

Stacked Bar Chart
=================

.. autofunction:: generate_stacked_bar_chart

   Generate horizontal stacked bar chart comparing feature changes.

   **Parameters:**

   - **factual_df** (pd.DataFrame): Factual data
   - **cf_df** (pd.DataFrame): Original counterfactual data
   - **ace_df** (pd.DataFrame): Refined counterfactual data
   - **label_column** (str): Name of label column
   - **save_path** (str, optional): Directory to save figure
   - **refined_color** (str): Color for refined bars (default: "#D9F2D0" green)
   - **counterfactual_color** (str): Color for original bars (default: "#FBE3D6" orange)
   - **instance_labels** (List[str], optional): Custom instance labels
   - **figsize** (Tuple[int, int], optional): Figure size
   - **dpi** (int, optional): Resolution

   **Visual Elements:**

   - Y-axis (rows): Each factual instance
   - X-axis: Percentage of feature changes
   - Green bar: Refined counterfactual modifications
   - Orange bar: Original counterfactual modifications

   **Returns:** matplotlib Figure object

.. autofunction:: create_stacked_bar_chart

   Low-level function for creating stacked bar charts.

   Used internally by generate_stacked_bar_chart.

Diversity Analysis
==================

.. autofunction:: generate_diversity_for_all_instances

   Find all minimal feature combinations for all instances.

   **Parameters:**

   - **factual_df** (pd.DataFrame): Factual data
   - **refined_counterfactual_df** (pd.DataFrame): Refined counterfactual data
   - **ml_model** (Model): Wrapped ML model
   - **label_column** (str): Name of label column
   - **cola_data** (COLAData, optional): Data object for transformations
   - **max_features** (int, optional): Maximum features to try

   **Algorithm:**

   1. For each instance, enumerate all feature subsets
   2. Test which subsets can flip the label
   3. Find minimal subsets (no proper subset also works)
   4. Return all minimal alternatives

   **Returns:** Tuple of (factual_df, List[styled_diversity_dataframes])

.. autofunction:: generate_diversity_dataframe

   Generate diversity DataFrame for a single instance.

   **Parameters:**

   - **factual_row** (pd.Series): Single factual instance
   - **refined_counterfactual_row** (pd.Series): Single refined CF
   - **minimal_combinations** (List[Set[str]]): Minimal feature combinations
   - **label_column** (str): Label column name

   **Returns:** pd.DataFrame with all minimal alternatives

.. autofunction:: highlight_diversity_changes

   Highlight diversity DataFrame with color coding.

   **Parameters:**

   - **factual_row** (pd.Series): Factual instance
   - **diversity_df** (pd.DataFrame): Diversity results
   - **label_column** (str): Label column name

   **Returns:** Styled DataFrame with highlighted changes

.. autofunction:: find_minimal_feature_combinations

   Find minimal feature combinations for single instance.

   **Parameters:**

   - **factual_row** (pd.Series): Single factual instance
   - **refined_counterfactual_row** (pd.Series): Single refined CF
   - **ml_model** (Model): Wrapped ML model
   - **label_column** (str): Label column name
   - **cola_data** (COLAData, optional): For transformations

   **Algorithm:**

   Uses exhaustive enumeration starting from 1 feature up to all changed features.
   Ensures true minimality: if changing feature A alone works, combinations
   like {A, B} are excluded.

   **Returns:** List[Set[str]] of minimal feature combinations

Examples
========

Direction Heatmap
-----------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_direction_heatmap

    fig = generate_direction_heatmap(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results',
        save_mode='combined'
    )

Via COLA Instance
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA

    cola = COLA(data=data, ml_model=ml_model)
    cola.set_policy(matcher="ot", attributor="pshap")
    cola.refine_counterfactuals(limited_actions=5)

    # Generate heatmap
    fig = cola.heatmap_direction(
        save_path='./results',
        save_mode='combined',
        show_axis_labels=True
    )

Binary Heatmap
--------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_binary_heatmap

    fig = generate_binary_heatmap(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results'
    )

Stacked Bar Chart
-----------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import generate_stacked_bar_chart

    fig = generate_stacked_bar_chart(
        factual_df=factual,
        cf_df=counterfactual,
        ace_df=refined_counterfactual,
        label_column='Risk',
        save_path='./results'
    )

Highlighted Comparison
----------------------

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import highlight_changes_comparison

    styled_ce, styled_ace = highlight_changes_comparison(
        factual_df=factual,
        cf1_df=counterfactual,
        cf2_df=refined_counterfactual,
        label_column='Risk'
    )

    # Display in Jupyter
    display(styled_ce)
    display(styled_ace)

    # Save to HTML
    styled_ce.to_html('original_cf.html')
    styled_ace.to_html('refined_ace.html')

Diversity Analysis
------------------

Find All Minimal Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import (
        generate_diversity_for_all_instances
    )

    diversity_results = generate_diversity_for_all_instances(
        factual_df=factual,
        ml_model=ml_model,
        target_class=0,
        max_features=5
    )

    # Results: dict mapping instance_id to list of feature combinations
    for instance_id, combinations in diversity_results.items():
        print(f"Instance {instance_id}:")
        for combo in combinations:
            print(f"  - {combo}")

Highlighted Diversity DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.visualization import highlight_diversity_changes

    # Get highlighted diversity DataFrame
    diversity_styled = highlight_diversity_changes(
        factual_df=factual,
        diversity_results=diversity_results,
        label_column='Risk'
    )

    display(diversity_styled)

Complete Visualization Suite
-----------------------------

Generate All Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from xai_cola import COLA

    # Setup and refine
    cola = COLA(data=data, ml_model=ml_model)
    cola.set_policy(matcher="ot", attributor="pshap", random_state=42)
    cola.refine_counterfactuals(limited_actions=5)

    # Create output directory
    os.makedirs('./results', exist_ok=True)

    # Generate all visualizations
    cola.heatmap_direction(save_path='./results', save_mode='both')
    cola.heatmap_binary(save_path='./results', save_mode='both')
    cola.stacked_bar_chart(save_path='./results')

    # Highlighted DataFrames
    factual_style, ce_style, ace_style = cola.highlight_changes_final()
    ce_style.to_html('./results/original_cf.html')
    ace_style.to_html('./results/refined_ace.html')

    print("✓ All visualizations saved to ./results/")

Customization
=============

Figure Size and DPI
-------------------

.. code-block:: python

    # High resolution for publications
    fig = cola.heatmap_direction(
        save_path='./results',
        figsize=(14, 10),
        dpi=300
    )

    # Large size for presentations
    fig = cola.stacked_bar_chart(
        save_path='./results',
        figsize=(18, 10),
        dpi=150
    )

Axis Labels
-----------

.. code-block:: python

    # Show feature and instance names
    fig = cola.heatmap_direction(
        save_path='./results',
        show_axis_labels=True
    )

    # Hide for cleaner look
    fig = cola.heatmap_direction(
        save_path='./results',
        show_axis_labels=False
    )

Save Modes
----------

.. code-block:: python

    # Combined: CE and ACE side by side
    cola.heatmap_direction(save_mode='combined')

    # Separate: Two individual heatmaps
    cola.heatmap_direction(save_mode='separate')

    # Both: Generate both combined and separate
    cola.heatmap_direction(save_mode='both')

Multiple Output Formats
-----------------------

.. code-block:: python

    # Generate figure
    fig = cola.heatmap_direction(save_path='./results')

    # Save in multiple formats
    fig.savefig('./results/heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig('./results/heatmap.pdf', bbox_inches='tight')
    fig.savefig('./results/heatmap.svg', bbox_inches='tight')

Color Schemes
-------------

For direction heatmaps, colors indicate:

- 🟦 **Blue** - Feature value increased
- 🟧 **Red/Orange** - Feature value decreased
- ⬜ **White** - No change

For binary heatmaps:

- ⬛ **Black** - Feature changed
- ⬜ **White** - No change

For highlighted DataFrames:

- 🟦 **Blue background** - Value increased
- 🟧 **Orange background** - Value decreased
- ⬜ **White background** - No change

See Also
========

- :doc:`../user_guide/visualization` - Detailed visualization guide
- :doc:`cola` - COLA main class
- :doc:`ce_generator` - Counterfactual generators
