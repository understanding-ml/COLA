==================
Matching Policies
==================

Overview
========

Matching policies determine how COLA pairs factual instances with counterfactual instances before refinement. The choice of matching policy affects both the quality and computational cost of the refined counterfactuals.

COLA provides four matching strategies:

1. **Exact matching (ECT)** - Fast, deterministic matching (recommended)
2. **Optimal Transport (OT)** - Globally optimal matching (recommended)
3. **Nearest Neighbor (NN)** - Simple proximity-based matching (recommended)
4. **Coarsened Exact Matching (CEM)** - Coarsened exact matching

Quick Start
===========

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Initialize COLA
    sparsifier = COLA(data=data, ml_model=ml_model)

    # Set matching policy
    sparsifier.set_policy(
        matcher="ot",         # Matching strategy
        attributor="pshap",   # Feature attribution method
        random_state=42       # For reproducibility
    )

    # Query minimum actions needed
    min_actions = sparsifier.query_minimum_actions()

    # Refine counterfactuals
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

Matching Strategies
===================

1. Exact Matching (ECT)
--------------------------------

**When to use:**

- You need fast results
- You have clear class transitions (e.g., 0→1, 1→0)
- Number of factuals equals number of counterfactuals (n factuals = n counterfactuals)
- One-to-one matching is desired (creating an n×n identity matrix)

**How it works:**

Matches factual instances to counterfactuals based on exact class transitions. For instance, factuals with class 0 are matched to counterfactuals with class 1.

.. code-block:: python

    sparsifier.set_policy(
        matcher="ect",
        attributor="pshap"
    )

**Advantages:**

- ✅ Very fast
- ✅ Deterministic results
- ✅ Simple and interpretable
- ✅ No hyperparameters

**Disadvantages:**

- ⚠️ May not be globally optimal
- ⚠️ Requires balanced classes
- ⚠️ Limited flexibility

**Best for:** Binary classification with similar class sizes.

2. Optimal Transport (OT)
-------------------------

**When to use:**

- You want the best quality results
- Computational cost is acceptable
- You have many counterfactuals per instance

**How it works:**

Solves an optimal transport problem to find the globally optimal assignment between factual and counterfactual instances, minimizing total transportation cost.

.. code-block:: python

    sparsifier.set_policy(
        matcher="ot",
        attributor="pshap"
    )

**Advantages:**

- ✅ Globally optimal matching
- ✅ Best refinement quality
- ✅ Considers all possible pairings
- ✅ Theoretically grounded

**Disadvantages:**

- ⚠️ Slower than other methods
- ⚠️ Complexity: O(n³) for n instances


3. Nearest Neighbor (NN)
-------------------------

**When to use:**

- You want the simplest approach
- Computational resources are very limited
- Quick prototyping

**How it works:**

Matches each factual to its nearest counterfactual in feature space using Euclidean distance.

.. code-block:: python

    sparsifier.set_policy(
        matcher="nn",
        attributor="pshap"
    )

**Advantages:**

- ✅ Extremely fast
- ✅ Simple to understand
- ✅ Works with any data

**Disadvantages:**

- ⚠️ Locally optimal only
- ⚠️ Sensitive to scale

4. Coarsened Exact Matching (CEM)
----------------------------------

**When to use:**

- You want to match on coarsened/binned feature values
- Variables have natural stratification (e.g., age groups, income brackets)
- You need balance on important covariates
- Exact matching is too restrictive but you want interpretable strata

**How it works:**

Temporarily coarsens (bins) continuous variables into discrete strata, performs exact matching on these coarsened values, then uses original feature values for refinement. This balances the trade-off between exact matching precision and matching feasibility.

.. code-block:: python

    sparsifier.set_policy(
        matcher="cem",
        attributor="pshap"
    )

**Advantages:**

- ✅ More flexible than exact matching
- ✅ Ensures balance on key covariates
- ✅ Interpretable stratification
- ✅ Reduces model dependence

**Disadvantages:**

- ⚠️ Requires choosing binning strategy
- ⚠️ May reduce sample size if strata are too fine
- ⚠️ Less optimal than OT for complex relationships

Feature Attribution
===================

PSHAP Attributor
----------------

COLA uses PSHAP for feature attribution, determining which features are most important for the transition from factual to counterfactual.

.. code-block:: python

    sparsifier.set_policy(
        matcher="ot",
        attributor="pshap",
        random_state=42
    )

**How PSHAP works:**

1. For each factual-counterfactual pair, compute Shapley values
2. Rank features by their contribution to the class change
3. Select top-k features with highest importance
4. Generate refined counterfactual using only these features

**Parameters:**

- ``random_state`` (int): Random seed for reproducibility

Querying Minimum Actions
=========================

Before refining, you can query the minimum number of actions needed:

.. code-block:: python

    # Query minimum actions
    min_actions = sparsifier.query_minimum_actions()
    print(f"Minimum actions needed: {min_actions}")

    # Use this value for refinement
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

This tells you the theoretical minimum number of feature changes needed for your dataset.

Refinement Options
==================

Basic Refinement
----------------

.. code-block:: python

    # Refine with specific action limit
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

With Feature Restrictions
--------------------------

You can restrict which features can be modified:

.. code-block:: python

    # Only allow these features to change
    refined = sparsifier.refine_counterfactuals(
        limited_actions=5,
        features_to_vary=['Income', 'Duration', 'LoanAmount']
    )

.. note::
    This is different from the explainer's ``features_to_vary``. The explainer controls CF generation, while this controls CF refinement.

Getting All Results
-------------------

Get factual, original counterfactual, and refined counterfactual together:

.. code-block:: python

    factual_df, ce_df, ace_df = sparsifier.get_all_results(limited_actions=5)

    print("Original CF actions:", (factual_df != ce_df).sum().sum())
    print("Refined ACE actions:", (factual_df != ace_df).sum().sum())

Complete Examples
=================

Example 1: Optimal Transport with Minimum Actions
--------------------------------------------------

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE

    # Setup
    data = COLAData(factual_data=df, label_column='Risk')
    ml_model = Model(model=trained_model, backend="sklearn")

    # Generate CFs
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2
    )
    data.add_counterfactuals(cf, with_target_column=True)

    # Refine with OT
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ot", attributor="pshap", random_state=42)

    # Find and use minimum actions
    min_actions = sparsifier.query_minimum_actions()
    refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

    print(f"Refined {len(refined)} counterfactuals")
    print(f"Using {min_actions} feature changes per instance")

Example 2: Fast ECT Matching
-----------------------------

.. code-block:: python

    # For quick results, use ECT
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ect", attributor="pshap")

    # ECT is much faster than OT
    import time
    start = time.time()
    refined = sparsifier.refine_counterfactuals(limited_actions=5)
    print(f"Refinement time: {time.time() - start:.2f}s")

Example 3: Comparing Matchers
------------------------------

.. code-block:: python

    import pandas as pd

    results = []

    for matcher in ["ect", "ot", "nn", "softcem"]:
        sparsifier = COLA(data=data, ml_model=ml_model)
        sparsifier.set_policy(matcher=matcher, attributor="pshap")

        min_actions = sparsifier.query_minimum_actions()
        refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

        # Count changes
        factual_df, ce_df, ace_df = sparsifier.get_all_results(
            limited_actions=min_actions
        )
        n_changes = (factual_df != ace_df).sum().sum()

        results.append({
            'Matcher': matcher,
            'Min Actions': min_actions,
            'Total Changes': n_changes
        })

    results_df = pd.DataFrame(results)
    print(results_df)

Example 4: With Feature Restrictions
-------------------------------------

.. code-block:: python

    # Scenario: Only financial features can change
    financial_features = ['Income', 'LoanAmount', 'Duration']

    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher="ot", attributor="pshap")

    refined = sparsifier.refine_counterfactuals(
        limited_actions=3,
        features_to_vary=financial_features
    )

    # Verify only financial features changed
    factual_df, _, ace_df = sparsifier.get_all_results(limited_actions=3)

    for col in factual_df.columns:
        if col not in financial_features + ['Risk']:
            assert (factual_df[col] == ace_df[col]).all(), f"{col} changed!"

    print("✓ Only financial features were modified")

Choosing the Right Policy
==========================

Decision Guide
--------------

.. code-block:: text

    ┌─────────────────────────────────────┐
    │  Need best quality?                 │
    │  ├─ Yes → Use OT                    │
    │  └─ No → Continue                   │
    └─────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │  Need fast results?                 │
    │  ├─ Yes → Use ECT                   │
    │  └─ No → Continue                   │
    └─────────────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │  Have complex overlaps?             │
    │  ├─ Yes → Use SoftCEM               │
    │  └─ No → Use NN                     │
    └─────────────────────────────────────┘

Recommendation Table
--------------------

+------------------+----------------+----------------+-----------------+
| Scenario         | Matcher        | Speed          | Quality         |
+==================+================+================+=================+
| Production use   | **OT**         | Medium         | Best            |
+------------------+----------------+----------------+-----------------+
| Quick iteration  | **ECT**        | Fast           | Good            |
+------------------+----------------+----------------+-----------------+
| Binary class     | **ECT**        | Fast           | Good            |
+------------------+----------------+----------------+-----------------+
| Large dataset    | **ECT/NN**     | Fast           | Acceptable      |
+------------------+----------------+----------------+-----------------+
| Research         | **OT**         | Medium         | Best            |
+------------------+----------------+----------------+-----------------+
| Prototype        | **NN**         | Very Fast      | Basic           |
+------------------+----------------+----------------+-----------------+

Common Issues
=============

Issue 1: Matching Takes Too Long
---------------------------------

**Problem:** OT matching is slow on large datasets.

**Solution:** Use ECT or NN for faster results:

.. code-block:: python

    # ❌ Slow on 1000+ instances
    sparsifier.set_policy(matcher="ot", attributor="pshap")

    # ✅ Much faster
    sparsifier.set_policy(matcher="ect", attributor="pshap")

Issue 2: Unbalanced Classes
----------------------------

**Problem:** CEM fails with unbalanced class distribution.

**Error:**

.. code-block:: text

    ValueError: Cannot match - unbalanced class distribution

**Solution:** Use OT which handles imbalance:

.. code-block:: python

    # ✅ Works with any class distribution
    sparsifier.set_policy(matcher="ot", attributor="pshap")

Issue 3: Inconsistent Results
------------------------------

**Problem:** Results vary between runs.

**Solution:** Set random_state for reproducibility:

.. code-block:: python

    # ✅ Reproducible results
    sparsifier.set_policy(
        matcher="ot",
        attributor="pshap",
        random_state=42  # Fixed seed
    )

Best Practices
==============

✅ **DO:**

1. **Start with ECT for exploration**

   .. code-block:: python

       # Quick first pass
       sparsifier.set_policy(matcher="ect", attributor="pshap")

2. **Use OT for final results**

   .. code-block:: python

       # Best quality for production
       sparsifier.set_policy(matcher="ot", attributor="pshap")

3. **Always set random_state for research**

   .. code-block:: python

       sparsifier.set_policy(
           matcher="ot",
           attributor="pshap",
           random_state=42
       )

4. **Query minimum actions before refining**

   .. code-block:: python

       min_actions = sparsifier.query_minimum_actions()
       refined = sparsifier.refine_counterfactuals(limited_actions=min_actions)

❌ **DON'T:**

1. **Don't use CEM as default when having few samples** - it's lowest quality

2. **Don't ignore computational cost** - OT can be slow on large datasets

3. **Don't forget to set the policy** - must call ``set_policy()`` before refinement

   .. code-block:: python

       # ❌ Error - no policy set
       sparsifier = COLA(data=data, ml_model=ml_model)
       refined = sparsifier.refine_counterfactuals(limited_actions=5)

       # ✅ Correct
       sparsifier.set_policy(matcher="ot", attributor="pshap")
       refined = sparsifier.refine_counterfactuals(limited_actions=5)

API Reference
=============

For complete parameter details, see:

- :class:`~xai_cola.ce_sparsifier.COLA`
- :class:`~xai_cola.ce_sparsifier.policies.matching.CounterfactualOptimalTransportPolicy`
- :class:`~xai_cola.ce_sparsifier.policies.feature_attributor.PSHAP`

Next Steps
==========

- Learn about :doc:`visualization` - Visualizing refinement results
- See :doc:`explainers` - Generating counterfactuals
- Review :doc:`data_interface` - Managing data
