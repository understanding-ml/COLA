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
      ~COLA.highlight_changes_final
      ~COLA.highlight_diversity_changes
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

.. automethod:: COLA.heatmap_direction

.. automethod:: COLA.stacked_bar_chart

.. automethod:: COLA.highlight_changes_final

.. automethod:: COLA.highlight_diversity_changes

Diversity Analysis
------------------

.. automethod:: COLA.diversity_analysis

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
