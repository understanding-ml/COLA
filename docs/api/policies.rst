==============
Policies API
==============

Overview
========

COLA's refinement process is built on three core policy components:

1. **Matching Policies** - Determine how to pair factual and counterfactual instances
2. **Feature Attribution** - Compute importance scores for each feature
3. **Data Composer** - Select and combine features based on importance

These policies work together to create sparse, actionable counterfactual explanations.

Matching Policies
=================

.. currentmodule:: xai_cola.ce_sparsifier.policies.matching

Matching policies define how factual instances are paired with counterfactual instances
to determine which features to modify.

Base Matcher
------------

.. autoclass:: BaseCounterfactualMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for all matching policies.

   **Key Methods:**

   - ``compute_prob_matrix_of_factual_and_counterfactual()`` - Returns (N, M) probability matrix
   - ``convert_matrix_to_policy()`` - Converts probability matrix to policy

Optimal Transport (Recommended)
--------------------------------

.. autoclass:: CounterfactualOptimalTransportPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Globally optimal matching using optimal transport (Wasserstein distance).

   **Algorithm:** Solves Earth Mover's Distance (EMD) or Sinkhorn problem to find
   the optimal soft assignment between factual and counterfactual distributions.

   **Advantages:**

   - Globally optimal matching
   - Supports n-to-m matching (different instance counts)
   - Theoretically grounded
   - Robust to outliers

   **Disadvantages:**

   - Slower than other methods (O(n³) for EMD, O(n²) for Sinkhorn)

   **Best for:** Most use cases, especially when you want high-quality results

   **Policy String:** ``"ot"``

Exact Class Transition
-----------------------

.. autoclass:: CounterfactualExactMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   One-to-one deterministic matching (identity matrix).

   **Algorithm:** Requires equal number of factual and counterfactual instances.
   Matches i-th factual to i-th counterfactual.

   **Advantages:**

   - Fastest method (O(1))
   - Deterministic and reproducible
   - Simple to understand

   **Disadvantages:**

   - Requires N_factual == N_counterfactual
   - No optimization, may not be ideal matching

   **Best for:** When counterfactuals are generated pairwise (1 CF per factual)

   **Policy String:** ``"ect"``

Nearest Neighbor
----------------

.. autoclass:: CounterfactualNearestNeighborMatchingPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Simple proximity-based matching using Euclidean distance.

   **Algorithm:** For each factual instance, find the nearest counterfactual
   instance in feature space.

   **Advantages:**

   - Fast (O(n²))
   - Supports n-to-m matching
   - Intuitive (nearest neighbor)

   **Disadvantages:**

   - Local greedy choice, not globally optimal
   - Sensitive to feature scales

   **Best for:** Quick experiments, baseline comparisons

   **Policy String:** ``"nn"``

Soft CEM (Coarsened Exact Matching)
------------------------------------

.. autoclass:: CounterfactualSoftCEMPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Probabilistic soft matching based on Coarsened Exact Matching.

   **Algorithm:** Groups instances into strata based on similar features,
   then uses optimal transport within and across strata.

   **Key Parameters:**

   - ``treatment_name`` (str): Treatment column name (default: "T")
   - ``scale`` (bool): Apply StandardScaler normalization (default: True)
   - ``use_sinkhorn`` (bool): Use Sinkhorn vs EMD (default: False)
   - ``sinkhorn_eps`` (float): Entropy regularization (default: 0.05)
   - ``cost_clip_quantile`` (float): Cost clipping quantile (default: 0.9)

   **Advantages:**

   - Balances groups by confounders
   - Good for causal inference tasks
   - Provides matching quality diagnostics

   **Disadvantages:**

   - More complex configuration
   - Slower than NN

   **Best for:** Causal explanation tasks, when you want balanced matching

   **Policy String:** ``"cem"`` or ``"softcem"``

Feature Attribution
===================

.. currentmodule:: xai_cola.ce_sparsifier.policies.feature_attributor

Feature attribution determines which features are most important for the
class transition between factual and counterfactual instances.

PSHAP (Pair-wise SHAP)
-----------------------

.. autoclass:: PSHAP
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Pair-wise SHAP for feature importance attribution.

   **Algorithm:** Computes SHAP (SHapley Additive exPlanations) values for each
   factual-counterfactual pair, weighted by the joint probability from the matching policy.

   **How it works:**

   1. For each factual instance, use the matched counterfactuals as baseline
   2. Compute SHAP values weighted by matching probabilities
   3. Rank features by absolute SHAP value (importance)

   **Key Parameters:**

   - ``ml_model`` (Model): Wrapped ML model
   - ``x_factual`` (np.ndarray): Factual instances (N, P)
   - ``x_counterfactual`` (np.ndarray): Counterfactual instances (M, P)
   - ``joint_prob`` (np.ndarray): Matching probability matrix (N, M)
   - ``random_state`` (int): Random seed for SHAP sampling
   - ``feature_names`` (List[str], optional): Feature names for debugging

   **Returns:**

   - ``varphi`` (np.ndarray): Feature importance matrix (N, P)

   **Mathematical Foundation:**

   For each factual instance i, PSHAP computes:

   .. math::

       \\varphi_{i,j} = \\sum_{k=1}^{M} P(CF_k | F_i) \\cdot SHAP_j(F_i, CF_k)

   where P(CF_k | F_i) comes from the matching policy.

   **Advantages:**

   - Theoretically grounded (Shapley values)
   - Model-agnostic
   - Considers interactions between features
   - Weighted by matching quality

   **Policy String:** ``"pshap"``

Data Composer
=============

.. currentmodule:: xai_cola.ce_sparsifier.policies.data_composer

The data composer selects which features to modify based on importance scores.

DataComposer Class
------------------

.. autoclass:: DataComposer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Composes refined counterfactuals by selecting maximum-weight matches.

   **Algorithm:** For each factual instance, select the counterfactual with
   the highest matching probability (max-weight method).

   **How it works:**

   .. code-block:: python

       for each factual instance i:
           j = argmax(joint_prob[i, :])
           refined_cf[i] = counterfactual[j]

   **Key Methods:**

   - ``calculate_q()`` - Returns (N, P) matrix where each row is the max-weight CF

   **Advantages:**

   - Preserves exact counterfactual instances
   - Simple and deterministic
   - Fast (O(n))

   **Note:** The actual feature selection (limiting actions) happens in COLA,
   which uses the importance scores from PSHAP to decide which features to keep.

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
