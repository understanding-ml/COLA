============================
Counterfactual Explainers
============================

Overview
========

COLA provides built-in counterfactual explanation generators and supports integration with external explainers. These explainers generate the initial counterfactuals that COLA then refines to minimize the number of required actions.

Built-in Explainers
===================

COLA includes two main explainers:

1. **DiCE** - Instance-wise counterfactual generation (Diverse Counterfactual Explanations) - `FaccT 2020 paper <https://arxiv.org/abs/1905.07697>`_
2. **DisCount** - Distributional counterfactual generation (Distributional Counterfactual Explanations With Optimal Transport) - `AISTATS 2025 oral <https://arxiv.org/pdf/2401.13112>`_

DiCE Explainer
==============

DiCE generates multiple diverse counterfactuals for individual instances. For more
details, please refer to https://interpret.ml/DiCE/.

**When to use DiCE:**

- You want diverse counterfactual options for each instance
- You need instance-level explanations
- You want to control which features can be changed
- You care about proximity and sparsity

Basic Usage of our Built-in DiCE.
-------------------------------------

.. code-block:: python

    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Prepare data
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )

    # 2. Wrap model
    ml_model = Model(model=trained_model, backend="sklearn")

    # 3. Create explainer
    explainer = DiCE(ml_model=ml_model)

    # 4. Generate counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,        # Generate CFs for instances with class 1
        total_cfs=2,            # Generate 2 CFs per instance
        continuous_features=['Age', 'Income']
    )

    # 5. Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

Parameters Explained
--------------------

**factual_class** (*int*)
    The class label of factual instances. The explainer will generate counterfactuals for the opposite class.

    .. code-block:: python

        # Factual instances have class 1, generate CFs for class 0
        factual_class=1

        # Factual instances have class 0, generate CFs for class 1
        factual_class=0

**total_cfs** (*int*)
    Number of counterfactuals per instance.

    .. code-block:: python

        total_cfs=1   # One CF per instance (faster)
        total_cfs=5   # Five CFs per instance (more diverse)

**continuous_features** (*list*)
    Features that can take continuous values.

    .. code-block:: python

        continuous_features=['Age', 'Income', 'Duration']

**features_to_keep** (*list*)
    Features that should NOT be changed (immutable features).

    .. code-block:: python

        # Don't change Age or Gender
        features_to_keep=['Age', 'Gender']

**features_to_vary** (*list*)
    Only these features can be changed.

    .. code-block:: python

        # Only allow changing Income and Duration
        features_to_vary=['Income', 'Duration']

    .. note::
        Use either ``features_to_keep`` OR ``features_to_vary``, not both.

**permitted_range** (*dict*)
    Allowed ranges for features. For numerical features, specify [min, max]. For categorical features, specify list of allowed values.

    .. code-block:: python

        permitted_range={
            'age': [20, 30],                              # Age must be between 20-30
            'education': ['Doctorate', 'Prof-school']     # Only these education levels allowed
        }

Advanced DiCE Usage
-------------------

Example 1: With Immutable Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Age and Gender cannot be changed
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=3,
        features_to_keep=['Age', 'Gender'],
        continuous_features=['Income', 'Duration']
    )

Example 2: With Feature Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Realistic constraints on features
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income', 'LoanAmount'],
        permitted_range={
            'Age': [18, 70],
            'Income': [10000, 200000],
            'LoanAmount': [1000, 50000]
        }
    )

Example 3: Selective Feature Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Scenario: Only allow financial features to change
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        features_to_vary=['Income', 'LoanAmount', 'Duration'],
        continuous_features=['Income', 'LoanAmount', 'Duration']
    )

DisCount Explainer
==================

DisCount generates distributional counterfactuals - it finds a counterfactual distribution that maintains similar structure to the factual distribution while achieving the desired prediction outcome.

**When to use DisCount:**

- You have groups of instances to explain
- You care about distributional properties
- You want cost-efficient group-level changes
- You need to maintain data distribution shape

**Requirements:**

- Only compatible with PyTorch models (``backend='pytorch'``)
- Model must be wrapped with ``Model(model=..., backend='pytorch')``

Mathematical Background
-----------------------

DisCount solves a constrained optimization problem to find a counterfactual distribution **x** that is close to the factual distribution **x'** while ensuring the model predictions **y = b(x)** are close to a target distribution **y\***.

**Optimization Objective:**

The algorithm minimizes a weighted objective function:

.. math::

    Q = (1 - \eta) Q_x(x, \mu) + \eta Q_y(x, \nu)

where:

- **Q_x(x, μ)**: Sliced Wasserstein distance between counterfactual and factual feature distributions
- **Q_y(x, ν)**: Wasserstein distance between counterfactual predictions and target distribution
- **η ∈ [l, r]**: Balancing weight that trades off between input similarity and output accuracy

**Constraints:**

The optimization is subject to distributional constraints:

.. math::

    \text{SW}_2(x, x') &\leq U_1 \quad \text{(input distributional proximity)} \\
    \text{W}_2(b(x), y^*) &\leq U_2 \quad \text{(output prediction accuracy)}

**Key Components:**

1. **Sliced Wasserstein Distance (SW₂)**: Computed by projecting distributions onto random 1D directions Θ = {θₖ}ₖ₌₁ᴺ and averaging the 1D Wasserstein distances
2. **Trimmed Distance**: Uses trimming constant δ to remove outliers from both tails of the distribution for robust computation
3. **Unified Confidence Upper Bound (UCL)**: Statistical upper bounds on W₂ and SW₂ with Bonferroni correction for the n_proj projections
4. **Interval Narrowing**: Adaptive algorithm to find optimal balancing weight η based on constraint slack: a = U₁ - SW₂, b = U₂ - W₂

Basic Usage
-----------

.. code-block:: python

    from xai_cola.ce_generator import DisCount
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Prepare data
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Credit amount', 'Duration']
    )

    # 2. Wrap PyTorch model
    ml_model = Model(model=pytorch_model, backend="pytorch")

    # 3. Create explainer
    explainer = DisCount(ml_model=ml_model)

    # 4. Generate distributional counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,           # Factual instances class
        lr=5e-2,                   # Learning rate
        n_proj=10,                 # Number of projections
        delta=0.15,                # Trimming constant
        U_1=0.4,                   # Distributional distance bound
        U_2=0.02,                  # Prediction distance bound
        max_iter=100,              # Maximum iterations
        silent=False               # Print optimization logs
    )

    # 5. Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

Parameters Explained
--------------------

**factual_class** (*int*, default=1)
    The class label of factual instances. The explainer will generate counterfactuals for the opposite class.

**lr** (*float*, default=5e-2)
    Learning rate for optimization. Controls the step size in gradient descent updates.

    .. code-block:: python

        lr=5e-2   # Default: moderate learning rate
        lr=1e-1   # Faster convergence, may be unstable
        lr=1e-3   # Slower but more stable

**n_proj** (*int*, default=10)
    Number of random projections for computing sliced Wasserstein distance. More projections give better approximation but slower computation.

    .. code-block:: python

        n_proj=10   # Default: good balance
        n_proj=50   # More accurate, slower
        n_proj=5    # Faster, less accurate

**delta** (*float*, default=0.15)
    Trimming constant ∈ (0, 0.5) for robust distance computation. Trims delta fraction from both tails of the distribution when computing Wasserstein distance.

    .. code-block:: python

        delta=0.15  # Default: moderate robustness
        delta=0.05  # Less trimming, more sensitive to outliers
        delta=0.3   # More trimming, more robust to outliers

**U_1** (*float*, default=0.4)
    Upper bound for input distributional distance (sliced Wasserstein distance between factual and counterfactual feature distributions). Controls how much the counterfactual distribution can differ from the factual distribution.

    .. code-block:: python

        U_1=0.4   # Default: moderate similarity
        U_1=0.2   # Stricter - counterfactuals closer to factuals
        U_1=0.6   # Looser - allow more distribution shift

**U_2** (*float*, default=0.02)
    Upper bound for output prediction distance (Wasserstein distance between counterfactual predictions and target distribution). Controls how close the counterfactual predictions should be to the desired outcome.

    .. code-block:: python

        U_2=0.02  # Default: tight prediction control
        U_2=0.01  # Stricter - predictions closer to target
        U_2=0.05  # Looser - allow more prediction variance

**l** (*float*, default=0.15)
    Lower bound for interval narrowing in the balancing weight η search. Used in the optimization algorithm to balance input and output constraints.

    .. code-block:: python

        l=0.15  # Default lower bound
        # Typically kept in [0, 0.5] range

**r** (*float*, default=1)
    Upper bound for interval narrowing in the balancing weight η search.

    .. code-block:: python

        r=1     # Default upper bound
        # Typically kept at 1.0

**max_iter** (*int*, default=100)
    Maximum number of optimization iterations. More iterations allow better convergence but take longer.

    .. code-block:: python

        max_iter=100   # Default
        max_iter=200   # More iterations for complex problems
        max_iter=50    # Faster but may not converge

**tau** (*float*, default=1e1)
    Step size for manifold gradient descent. Controls the scale of gradient updates in the optimization.

    .. code-block:: python

        tau=1e1   # Default
        tau=1e2   # Larger steps, faster but may be unstable
        tau=1e0   # Smaller steps, more stable

**silent** (*bool*, default=False)
    Whether to suppress optimization logs during training.

    .. code-block:: python

        silent=False  # Print logs (useful for debugging)
        silent=True   # Suppress logs (cleaner output)

**explain_columns** (*list*, optional)
    List of column names that can be modified during optimization. If None, all transformed features can be modified.

    .. code-block:: python

        explain_columns=['Credit amount', 'Duration']  # Only modify these
        explain_columns=None  # Allow all features to change

Advanced DisCount Usage
-----------------------

Example 1: With Selective Feature Modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Only allow financial features to change
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        explain_columns=['Credit amount', 'Duration', 'Saving accounts'],
        lr=5e-2,
        U_1=0.3,  # Tighter distributional constraint
        U_2=0.01,  # Tighter prediction constraint
        max_iter=150
    )

Example 2: Faster Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Trade accuracy for speed
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        n_proj=5,       # Fewer projections
        max_iter=50,    # Fewer iterations
        silent=True,    # No logs
        lr=1e-1         # Larger learning rate
    )

Example 3: High-Quality Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Prioritize quality over speed
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        n_proj=50,      # More projections
        max_iter=300,   # More iterations
        delta=0.1,      # Less trimming
        U_1=0.2,        # Tighter distributional bound
        U_2=0.01,       # Tighter prediction bound
        lr=1e-2         # Smaller, more stable learning rate
    )

Algorithm Details
-----------------

**Parameter Correspondence with Theory:**

The implementation maps to the theoretical algorithm as follows:

**Core Optimization Variables:**

- **x**: Counterfactual feature distribution (optimized)
- **x'**: Factual feature distribution (fixed, from ``data``)
- **y = b(x)**: Model predictions on counterfactuals
- **y\***: Target prediction distribution (default: all instances predicted as ``1 - factual_class``)
- **b(·)**: Prediction model (from ``ml_model``)

**Distance Metrics:**

.. math::

    \text{SW}_2(x, x') = \frac{1}{N} \sum_{k=1}^{N} \text{W}_2(\theta_k^T x, \theta_k^T x')

where:

- N = ``n_proj``: Number of random projection directions
- θₖ: Random unit vectors sampled from unit sphere
- W₂: 1D Wasserstein-2 distance with δ-trimming

**Trimmed Wasserstein Distance:**

.. math::

    \text{W}_{2,\delta}(P, Q) = \frac{1}{1-2\delta} \int_{\delta}^{1-\delta} |F_P^{-1}(u) - F_Q^{-1}(u)|^2 du

where δ = ``delta`` removes outliers from distribution tails.

**Balancing Weight Update:**

The interval narrowing algorithm updates η at each iteration based on constraint slack:

- **a** = U₁ - SW₂(x, x'): Slack in input constraint
- **b** = U₂ - W₂(b(x), y\*): Slack in output constraint

The balancing weight η is computed as:

- If **a < 0** and **b ≥ 0**: η = l (prioritize input constraint)
- If **a ≥ 0** and **b < 0**: η = r (prioritize output constraint)
- If **a < 0** and **b < 0**: η = l + b/(a+b) × (r-l) (both constraints violated)
- If **a ≥ 0** and **b ≥ 0**: η = l + a/(a+b) × (r-l) (both constraints satisfied)

where [l, r] is the search interval (``l``, ``r`` parameters).

**Gradient Update:**

.. math::

    x^{(t+1)} = x^{(t)} - \tau \cdot \nabla_x Q

where:

- τ = ``tau``: Step size for manifold gradient descent
- The actual update uses SGD/Adam with learning rate = ``lr``

**Unified Confidence Upper Bound (UCL):**

The algorithm uses statistical upper bounds to ensure constraints hold with high probability:

.. math::

    P(\text{SW}_2 \leq \text{UCL}_x) &\geq 1 - \alpha/2 \\
    P(\text{W}_2 \leq \text{UCL}_y) &\geq 1 - \alpha/2

where UCL uses δ-trimming and Bonferroni correction (α/N for N projections).

**Hyperparameter Tuning Recommendations:**

1. **n_proj** ∝ feature dimension: For d-dimensional features, use n_proj ≈ 10-50
2. **delta** ∈ [0.05, 0.3]: Smaller for clean data, larger for noisy data
3. **U_1/U_2** from domain knowledge: Analyze historical distribution shifts or business tolerance
4. **lr ≈ tau**: In practice, set both to similar magnitudes (e.g., 5e-2 and 1e1)
5. **max_iter**: 100-300 for convergence; use early stopping based on constraint satisfaction

External Explainers
===================

You can use COLA with any counterfactual explainer that produces DataFrames or arrays counterfactuals.

.. code-block:: python

    # Your custom explainer
    def my_explainer(X, model):
        # ... your counterfactual generation logic ...
        return counterfactuals_df

    # Generate CFs
    cf_df = my_explainer(X_test, model)

    # Use with COLA
    data = COLAData(factual_data=X_test, label_column='y', numerical_features=['age', 'income'])
    data.add_counterfactuals(cf_df, with_target_column=True)

    # Refine with COLA
    from xai_cola import COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    refined = sparsifier.refine_counterfactuals(limited_actions=5)


Common Issues
=============

Issue 1: No Counterfactuals Found
----------------------------------

**Error:**

.. code-block:: text

    ValueError: No valid counterfactuals found

**Possible causes:**

1. **Too many immutable features** - relax ``features_to_keep``
2. **Too strict ranges** - widen ``permitted_range``
3. **Model is too confident** - increase ``total_cfs`` or adjust proximity weight

**Solutions:**

.. code-block:: python

    # ❌ Too restrictive
    explainer.generate_counterfactuals(
        data=data,
        features_to_keep=['Age', 'Gender', 'Income', 'Job'],  # Too many!
        permitted_range={'Duration': [1, 2]}  # Too narrow!
    )

    # ✅ More flexible
    explainer.generate_counterfactuals(
        data=data,
        features_to_keep=['Age', 'Gender'],  # Only truly immutable
        permitted_range={'Duration': [1, 60]}  # Reasonable range
    )


Issue 2: Shape Mismatch
------------------------

**Error:**

.. code-block:: text

    ValueError: Factual and counterfactual have different number of columns

**Cause:** Counterfactual DataFrame doesn't match factual structure.

**Solution:** Ensure column consistency:

.. code-block:: python

    # Check columns
    print("Factual columns:", data.factual_df.columns.tolist())
    print("CF columns:", cf_df.columns.tolist())

    # Make sure they match (order doesn't matter, but names must)
    assert set(data.factual_df.columns) == set(cf_df.columns)

Best Practices
==============

✅ **DO:**

1. **Start with fewer CFs** for faster iteration

   .. code-block:: python

       # Start with 1 CF per instance
       total_cfs=1

       # Increase later if needed
       total_cfs=5

2. **Always specify continuous_features**

   .. code-block:: python

       continuous_features=['Age', 'Income', 'Duration']

3. **Use realistic feature constraints**

   .. code-block:: python

       features_to_keep=['Age', 'Gender']  # Truly immutable
       permitted_range={'Income': [0, 500000]}  # Realistic bounds

4. **Verify counterfactuals before refinement**

   .. code-block:: python

       # Check CF predictions
       cf_preds = ml_model.predict(cf_df.drop('Risk', axis=1))
       print("CF predictions:", cf_preds)
       print("Desired class:", desired_class)

❌ **DON'T:**

1. **Don't use too many immutable features**
2. **Don't forget to add counterfactuals to COLAData**

   .. code-block:: python

       # ❌ Forgot this step
       factual, cf = explainer.generate_counterfactuals(...)
       sparsifier = COLA(data=data, ml_model=ml_model)  # Error!

       # ✅ Remember to add CFs
       data.add_counterfactuals(cf, with_target_column=True)
       sparsifier = COLA(data=data, ml_model=ml_model)  # Works!

3. **Don't mix continuous and categorical in continuous_features**

API Reference
=============

For complete parameter details, see:

- :class:`~xai_cola.ce_generator.DiCE`
- :class:`~xai_cola.ce_generator.DisCount`

Next Steps
==========

- Learn about :doc:`matching_policies` - Configuring COLA refinement
- Explore :doc:`visualization` - Visualizing results
- See :doc:`data_interface` - Managing data
