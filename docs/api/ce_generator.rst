==========================
Counterfactual Generators
==========================

.. currentmodule:: xai_cola.ce_generator

DiCE Explainer
==============

.. autoclass:: DiCE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Diverse Counterfactual Explanations (DiCE) generator.

   Generates multiple diverse counterfactuals for each factual instance.
   Based on the DiCE paper: https://arxiv.org/abs/1905.07697

   **Key Features:**

   - Instance-level counterfactual generation
   - Multiple diverse counterfactuals per instance
   - Support for immutable features and feature ranges
   - Compatible with sklearn models and pipelines

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DiCE.__init__
      ~DiCE.generate_counterfactuals

Constructor
-----------

.. automethod:: DiCE.__init__

Generation Methods
------------------

.. automethod:: DiCE.generate_counterfactuals

DisCount Explainer
==================

.. autoclass:: DisCount
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Distributional Counterfactual Explanations with Optimal Transport (DisCount) generator.

   Generates distributional counterfactuals that maintain similar structure to the factual
   distribution while achieving desired prediction outcomes. Uses optimal transport theory
   and Wasserstein distances for robust distributional matching.

   Based on the paper: https://arxiv.org/pdf/2401.13112 (AISTATS 2025 oral)

   **Key Features:**

   - Distribution-level counterfactual generation
   - Preserves distributional structure
   - Uses sliced Wasserstein distance for efficient computation
   - Robust to outliers with trimmed distances
   - Statistical guarantees via confidence upper bounds

   **Requirements:**

   - Only compatible with PyTorch models (``backend='pytorch'``)
   - Model must be wrapped with ``Model(model=..., backend='pytorch')``

   **Mathematical Foundation:**

   DisCount solves a constrained optimization problem:

   .. math::

       \min_x Q = (1-\eta) Q_x(x, \mu) + \eta Q_y(x, \nu)

   subject to:

   .. math::

       \text{SW}_2(x, x') &\leq U_1 \\
       \text{W}_2(b(x), y^*) &\leq U_2

   where:

   - **x**: Counterfactual distribution (optimized)
   - **x'**: Factual distribution (fixed)
   - **y = b(x)**: Model predictions
   - **y\***: Target prediction distribution
   - **SW₂**: Sliced Wasserstein-2 distance
   - **W₂**: Wasserstein-2 distance
   - **U₁, U₂**: Upper bounds on distributional distances

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~DisCount.__init__
      ~DisCount.generate_counterfactuals

Constructor
-----------

.. automethod:: DisCount.__init__

Generation Methods
------------------

.. automethod:: DisCount.generate_counterfactuals

Base Explainer
==============

.. autoclass:: BaseExplainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Base class for counterfactual explainers.

   Use this as a template if you want to implement custom explainers
   compatible with COLA.

Examples
========

Using DiCE
----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Setup
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )
    ml_model = Model(model=trained_model, backend="sklearn")

    # Create DiCE explainer
    explainer = DiCE(ml_model=ml_model)

    # Generate counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income']
    )

    # Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

With Immutable Features
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Don't change Age or Gender
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=3,
        features_to_keep=['Age', 'Gender'],
        continuous_features=['Income', 'Duration']
    )

With Feature Ranges
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set realistic bounds
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income'],
        permitted_range={
            'age': [20, 30],                              # Numerical range
            'education': ['Doctorate', 'Prof-school']     # Categorical values
        }
    )

Only Vary Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Only financial features can change
    factual, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        features_to_vary=['Income', 'LoanAmount', 'Duration'],
        continuous_features=['Income', 'LoanAmount', 'Duration']
    )

Using DisCount
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import DisCount
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # Setup with PyTorch model
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Credit amount', 'Duration']
    )
    ml_model = Model(model=pytorch_model, backend="pytorch")

    # Create DisCount explainer
    explainer = DisCount(ml_model=ml_model)

    # Generate distributional counterfactuals
    factual, counterfactual = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        lr=5e-2,                   # Learning rate
        n_proj=10,                 # Number of projections
        delta=0.15,                # Trimming constant
        U_1=0.4,                   # Distributional distance bound
        U_2=0.02,                  # Prediction distance bound
        max_iter=100,              # Maximum iterations
        silent=False               # Print optimization logs
    )

    # Add to data
    data.add_counterfactuals(counterfactual, with_target_column=True)

With Selective Feature Modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Fast Optimization
~~~~~~~~~~~~~~~~~

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

High-Quality Results
~~~~~~~~~~~~~~~~~~~~

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

Integration with COLA
---------------------

Complete Workflow with DiCE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_generator import DiCE
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Setup
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Income']
    )
    ml_model = Model(model=trained_model, backend="sklearn")

    # 2. Generate CFs with DiCE
    explainer = DiCE(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        total_cfs=2,
        continuous_features=['Age', 'Income']
    )

    # 3. Add to data
    data.add_counterfactuals(cf, with_target_column=True)

    # 4. Refine with COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher='ot', attributor='pshap')
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

    # 5. Visualize
    sparsifier.heatmap_direction(save_path='./results')

Complete Workflow with DisCount
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola import COLA
    from xai_cola.ce_generator import DisCount
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model

    # 1. Setup with PyTorch model
    data = COLAData(
        factual_data=df,
        label_column='Risk',
        numerical_features=['Age', 'Credit amount', 'Duration']
    )
    ml_model = Model(model=pytorch_model, backend="pytorch")

    # 2. Generate distributional CFs with DisCount
    explainer = DisCount(ml_model=ml_model)
    _, cf = explainer.generate_counterfactuals(
        data=data,
        factual_class=1,
        lr=5e-2,
        U_1=0.4,
        U_2=0.02,
        max_iter=100
    )

    # 3. Add to data
    data.add_counterfactuals(cf, with_target_column=True)

    # 4. Refine with COLA
    sparsifier = COLA(data=data, ml_model=ml_model)
    sparsifier.set_policy(matcher='nn', attributor='pshap')
    refined = sparsifier.refine_counterfactuals(limited_actions=5)

    # 5. Visualize
    sparsifier.heatmap_binary(save_path='./results')
    sparsifier.stacked_bar_chart(save_path='./results')

Using Custom Explainers
-----------------------

Custom Implementation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from xai_cola.ce_generator import BaseExplainer

    class MyCustomExplainer(BaseExplainer):
        """Custom counterfactual explainer."""

        def __init__(self, ml_model, **kwargs):
            super().__init__(ml_model)
            self.kwargs = kwargs

        def generate_counterfactuals(
            self,
            data,
            factual_class,
            **params
        ):
            """Generate counterfactuals using custom logic."""

            # Your counterfactual generation logic here
            factual_df = data.get_factual_all()

            # ... generate cf_df ...
            # Ensure cf_df has same columns as factual_df

            return factual_df, cf_df

    # Use it
    explainer = MyCustomExplainer(ml_model=ml_model)
    factual, cf = explainer.generate_counterfactuals(data=data, factual_class=1)

Parameters Reference
====================

Common Parameters
-----------------

**data** : COLAData
    Data container with factual instances.

**factual_class** : int
    Class label to generate counterfactuals for. The explainer generates
    counterfactuals for the opposite class (1 - factual_class).

**continuous_features** : list of str, optional
    Names of continuous/numerical features.

**features_to_keep** : list of str, optional
    Features that should not be changed (immutable).

**features_to_vary** : list of str, optional
    Only these features can be changed.

DiCE-Specific Parameters
------------------------

**total_cfs** : int, default=1
    Number of counterfactuals to generate per instance.

**permitted_range** : dict, optional
    Allowed ranges for features.

    - For numerical features: ``{'feature': [min, max]}``
    - For categorical features: ``{'feature': ['value1', 'value2']}``

**diversity_weight** : float, optional
    Weight for diversity in generated counterfactuals.

**proximity_weight** : float, optional
    Weight for proximity to factual instance.

DisCount-Specific Parameters
-----------------------------

**lr** : float, default=5e-2
    Learning rate for optimization. Controls the step size in gradient descent updates.

**n_proj** : int, default=10
    Number of random projections for computing sliced Wasserstein distance.
    More projections give better approximation but slower computation.

**delta** : float, default=0.15
    Trimming constant ∈ (0, 0.5) for robust distance computation.
    Trims delta fraction from both tails of the distribution.

**U_1** : float, default=0.4
    Upper bound for input distributional distance (sliced Wasserstein distance
    between factual and counterfactual feature distributions).

**U_2** : float, default=0.02
    Upper bound for output prediction distance (Wasserstein distance between
    counterfactual predictions and target distribution).

**l** : float, default=0.15
    Lower bound for interval narrowing in the balancing weight η search.

**r** : float, default=1
    Upper bound for interval narrowing in the balancing weight η search.

**max_iter** : int, default=100
    Maximum number of optimization iterations.

**tau** : float, default=1e1
    Step size for manifold gradient descent.

**silent** : bool, default=False
    Whether to suppress optimization logs during training.

**explain_columns** : list of str, optional
    List of column names that can be modified during optimization.
    If None, all transformed features can be modified.

Algorithm Comparison
====================

+------------------+------------------+------------------------+
| Aspect           | DiCE             | DisCount               |
+==================+==================+========================+
| Level            | Instance-wise    | Distribution-wise      |
+------------------+------------------+------------------------+
| Backend          | Sklearn/PyTorch  | PyTorch only           |
+------------------+------------------+------------------------+
| Best for         | Individual       | Group explanations     |
|                  | explanations     |                        |
+------------------+------------------+------------------------+
| Diversity        | High (multiple   | Moderate (maintains    |
|                  | CFs per instance)| distribution)          |
+------------------+------------------+------------------------+
| Computational    | Fast             | Slower (optimization)  |
| Cost             |                  |                        |
+------------------+------------------+------------------------+
| Distributional   | No               | Yes (preserves         |
| Guarantees       |                  | structure)             |
+------------------+------------------+------------------------+
| Robustness       | Standard         | High (trimmed          |
|                  |                  | distances)             |
+------------------+------------------+------------------------+

See Also
========

- :doc:`../user_guide/explainers` - Detailed explainer guide with examples
- :doc:`cola` - COLA refinement API
- :doc:`data` - Data interface API
- :doc:`models` - Model interface API
