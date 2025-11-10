==============
Policies API
==============

Matching Policies
=================

.. currentmodule:: xai_cola.ce_sparsifier.policies.matching

Base Matcher
------------

.. autoclass:: BaseCounterfactualMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:

Optimal Transport
-----------------

.. autoclass:: CounterfactualOptimalTransportPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Globally optimal matching using optimal transport.

   Solves an optimal transport problem to find the best assignment
   between factual and counterfactual instances.

Exact Class Transition
-----------------------

.. autoclass:: CounterfactualExactMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Fast deterministic matching based on exact class transitions.

   Matches factual instances to counterfactuals with the desired
   target class.

Soft CEM
--------

.. autoclass:: CounterfactualSoftCEMPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Probabilistic soft matching based on CEM (Contrastive Explanation Method).

Nearest Neighbor
----------------

.. autoclass:: CounterfactualNearestNeighborMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Simple proximity-based matching using nearest neighbors.

Feature Attribution
===================

.. currentmodule:: xai_cola.ce_sparsifier.policies.feature_attributor

PSHAP Attributor
----------------

.. autoclass:: PSHAP
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Pair-wise SHAP (PSHAP) for feature attribution.

   Computes Shapley values for each factual-counterfactual pair to
   determine which features are most important for the class transition.

Data Composer
=============

.. currentmodule:: xai_cola.ce_sparsifier.policies.data_composer

.. autoclass:: DataComposer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Composes refined counterfactuals by selecting top-k features.

Examples
========

Using Matching Policies
------------------------

Optimal Transport
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA

    cola = COLA(data=data, ml_model=ml_model)
    cola.set_policy(matcher="ot", attributor="pshap")

    # OT provides globally optimal matching
    refined = cola.refine_counterfactuals(limited_actions=5)

Exact Class Transition
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fast matching for binary classification
    cola.set_policy(matcher="ect", attributor="pshap")
    refined = cola.refine_counterfactuals(limited_actions=5)

Nearest Neighbor
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simplest and fastest
    cola.set_policy(matcher="nn", attributor="pshap")
    refined = cola.refine_counterfactuals(limited_actions=5)

Soft CEM
~~~~~~~~

.. code-block:: python

    # Probabilistic soft matching
    cola.set_policy(matcher="softcem", attributor="pshap")
    refined = cola.refine_counterfactuals(limited_actions=5)

Direct Policy Usage
-------------------

Advanced: Using Policies Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_sparsifier.policies.matching import (
        CounterfactualOptimalTransportPolicy
    )
    from xai_cola.ce_sparsifier.policies.feature_attributor import PSHAP
    from xai_cola.ce_sparsifier.policies.data_composer import DataComposer

    # Create policy instances
    matcher = CounterfactualOptimalTransportPolicy()
    attributor = PSHAP(ml_model=ml_model, random_state=42)
    composer = DataComposer()

    # Use directly
    matching = matcher.match(factual_df, cf_df)
    importance = attributor.attribute(factual_df, cf_df, matching)
    refined_cf = composer.compose(factual_df, cf_df, importance, k=5)

Comparing Policies
------------------

.. code-block:: python

    import pandas as pd
    import time

    results = []

    for matcher_name in ["ect", "nn", "ot", "softcem"]:
        cola = COLA(data=data, ml_model=ml_model)
        cola.set_policy(matcher=matcher_name, attributor="pshap")

        # Measure time
        start = time.time()
        refined = cola.refine_counterfactuals(limited_actions=5)
        elapsed = time.time() - start

        # Count changes
        factual, ce, ace = cola.get_all_results(limited_actions=5)
        n_changes = (factual != ace).sum().sum()

        results.append({
            'Matcher': matcher_name,
            'Time (s)': elapsed,
            'Total Changes': n_changes
        })

    results_df = pd.DataFrame(results)
    print(results_df)

See Also
========

- :doc:`../user_guide/matching_policies` - Detailed policy guide
- :doc:`cola` - COLA main class
- :doc:`ce_generator` - Counterfactual generators
