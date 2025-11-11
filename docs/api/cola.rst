===============
COLA API
===============

.. currentmodule:: xai_cola.ce_sparsifier

COLA Class
==========

.. autoclass:: COLA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Main class for refining counterfactual explanations with limited actions.

   COLA (COunterfactual with Limited Actions) orchestrates the entire workflow of
   refining counterfactual explanations by limiting the number of feature changes while
   maintaining prediction flips.

   **Core Workflow:**

   1. Initialize with factual and counterfactual data
   2. Set matching and attribution policy
   3. Query minimum actions needed (optional)
   4. Refine counterfactuals with action limit
   5. Visualize and analyze results

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~COLA.__init__
      ~COLA.set_policy
      ~COLA.query_minimum_actions
      ~COLA.refine_counterfactuals
      ~COLA.get_refined_counterfactual
      ~COLA.get_all_results
      ~COLA.heatmap_binary
      ~COLA.heatmap_direction
      ~COLA.stacked_bar_chart
      ~COLA.highlight_changes_comparison
      ~COLA.highlight_changes_final
      ~COLA.diversity
      ~COLA.diversity_analysis

Constructor
-----------

.. automethod:: COLA.__init__

Policy Configuration
--------------------

.. automethod:: COLA.set_policy

Query Methods
-------------

.. automethod:: COLA.query_minimum_actions

Refinement Methods
------------------

.. automethod:: COLA.refine_counterfactuals

.. automethod:: COLA.get_refined_counterfactual

.. automethod:: COLA.get_all_results

Visualization Methods
---------------------

.. automethod:: COLA.heatmap_binary

   Generate binary heatmap showing which features changed.

   **Color Coding:**

   - Light grey: Unchanged features
   - Red: Changed features
   - Dark blue: Label column (prediction flip)

   **Returns:** Tuple of (counterfactual_fig, refined_counterfactual_fig)

.. automethod:: COLA.heatmap_direction

   Generate directional heatmap showing how features changed.

   **Color Coding:**

   - Light blue (#56B4E9): Numerical feature increased
   - Light cyan (#7FCDCD): Numerical feature decreased
   - Peru (#CD853F): Categorical feature changed
   - Black: Label column (prediction flip)
   - Light grey: Unchanged

   **Returns:** Tuple of (counterfactual_fig, refined_counterfactual_fig)

.. automethod:: COLA.stacked_bar_chart

   Generate horizontal stacked bar chart comparing feature changes.

   **Visual Elements:**

   - Y-axis (rows): Each factual instance
   - X-axis: Percentage of feature changes
   - Green bar: Refined counterfactual modifications
   - Orange bar: Original counterfactual modifications

   Shows that refined counterfactuals require fewer changes.

   **Returns:** Figure object

.. automethod:: COLA.highlight_changes_comparison

   Highlight DataFrame with format "old → new" showing changes.

   **Returns:** Tuple of (factual_styled, counterfactual_styled, refined_styled)

.. automethod:: COLA.highlight_changes_final

   Highlight DataFrame showing only final values with color coding.

   **Returns:** Tuple of (factual_styled, counterfactual_styled, refined_styled)

Diversity Analysis
------------------

.. automethod:: COLA.diversity

   Find all minimal feature combinations that achieve label flip.

   Uses exhaustive enumeration to find alternative minimal paths.
   Ensures true minimality: if one feature alone works, combinations
   with that feature are excluded.

   **Returns:** Tuple of (factual_df, List[styled_dataframes])

.. automethod:: COLA.diversity_analysis

   Perform diversity analysis for all instances.

   **Returns:** Dictionary mapping instance indices to minimal feature combinations

Examples
========

Basic Usage
-----------

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Initialize
    data = COLAData(factual_data=df, label_column='Risk')
    ml_model = Model(model=trained_model, backend="sklearn")

    # Generate counterfactuals (using any explainer)
    # ... cf_df = explainer.generate(...) ...
    data.add_counterfactuals(cf_df, with_target_column=True)

    # Create COLA instance
    cola = COLA(data=data, ml_model=ml_model)

    # Set refinement policy
    cola.set_policy(matcher='ot', attributor='pshap', random_state=42)

    # Refine counterfactuals
    refined_cf = cola.refine_counterfactuals(limited_actions=5)

    # Visualize
    cola.heatmap_direction(save_path='./results')
    cola.stacked_bar_chart(save_path='./results')

With Feature Restrictions
--------------------------

.. code-block:: python

    # Only allow certain features to change
    refined_cf = cola.refine_counterfactuals(
        limited_actions=5,
        features_to_vary=['Income', 'Duration', 'LoanAmount']
    )

Getting All Results
-------------------

.. code-block:: python

    # Get factual, original CF, and refined ACE
    factual_df, ce_df, ace_df = cola.get_all_results(limited_actions=5)

    # Compare
    print("Original changes:", (factual_df != ce_df).sum().sum())
    print("Refined changes:", (factual_df != ace_df).sum().sum())

See Also
========

- :doc:`../user_guide/matching_policies` - Guide on matching strategies
- :doc:`../user_guide/visualization` - Visualization guide
- :doc:`data` - Data interface documentation
- :doc:`models` - Model interface documentation
