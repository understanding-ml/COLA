==========================
Counterfactual Generators
==========================

.. currentmodule:: xai_cola.ce_generator

Module contents
===============

class **DiCE** (ml_model)

   **Bases:** ``CounterFactualExplainer``

   Diverse Counterfactual Explanations (DiCE) generator.

   Generates multiple diverse counterfactuals for each factual instance using the DiCE algorithm.
   DiCE creates counterfactual explanations by finding diverse alternative scenarios that would lead
   to a different prediction outcome.

   **Parameters:**
      **ml_model** (*Model*) -- The machine learning model wrapper.
      Should be created with ``Model(model=..., backend="sklearn")``.
      Can wrap:

      - Plain sklearn model (not recommended, preprocessing needed)
      - sklearn Pipeline with preprocessing (recommended)

   **References:**
      Mothilal, R. K., Sharma, A., & Tan, C. (2020).
      Explaining machine learning classifiers through diverse counterfactual explanations.
      In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency.
      https://doi.org/10.1145/3351095.3372850

   **Example:**

   .. code-block:: python

      from xai_cola.ce_generator import DiCE
      from xai_cola.ce_sparsifier.models import Model
      from xai_cola.ce_sparsifier.data import COLAData
      from sklearn.pipeline import Pipeline

      # Create pipeline with preprocessing
      pipe = Pipeline([("pre", column_transformer), ("clf", lgbm_clf)])
      ml_model = Model(model=pipe, backend="sklearn")

      # Initialize DiCE (no data needed)
      dice = DiCE(ml_model=ml_model)

      # Generate counterfactuals (provide data here)
      data = COLAData(factual_data=df, label_column='Risk')
      factual_df, cf_df = dice.generate_counterfactuals(
          data=data,
          continuous_features=['age', 'income'],
          total_cfs=5
      )

   **generate_counterfactuals** (data, factual_class=1, total_cfs=1, features_to_keep=None, continuous_features=None, permitted_range=None)

      Generate counterfactuals for the given factual data.

      **Parameters:**
         * **data** (*COLAData**, **required*) -- Data wrapper containing the factual data (original raw data).
           Must include both features and target column.
         * **factual_class** (*int**, **default=1*) -- The class of the factual data.
           Normally, we set the factual_class as 1 (positive class)
           and we hope the counterfactual is 0 (negative class).
         * **total_cfs** (*int**, **default=1*) -- Total number of counterfactuals required for each query instance.
         * **features_to_keep** (*list**, **optional*) -- List of feature names to keep unchanged in the counterfactuals.
           Uses original feature names (before any preprocessing).
         * **continuous_features** (*list**, **optional*) -- List of continuous/numerical feature names for dice_ml.Data.
           Uses original feature names (before any preprocessing).
           If None, will use all features as continuous features.
           Categorical features are automatically inferred as all features minus continuous_features.
         * **permitted_range** (*dict**, **optional*) -- Explicit feature range constraints.
           Format: ``{'feature_name': [min, max]}`` for numerical features,
           or ``{'feature_name': ['value1', 'value2']}`` for categorical features.

      **Returns:**
         * **factual_df** (*pd.DataFrame*) -- DataFrame with shape (n_samples, n_features + 1), includes target column.
           Contains the original factual data.
         * **counterfactual_df** (*pd.DataFrame*) -- DataFrame with shape (n_samples, n_features + 1), includes target column.
           Target column values are set based on actual model predictions.

      **Return type:**
         tuple

      **Raises:**
         **ValueError** -- If data is None

      **Example:**

      .. code-block:: python

         # Prepare data
         data = COLAData(factual_data=X_raw, label_column='target',
                        numerical_features=['age', 'income'])

         # Generate counterfactuals
         factual_df, cf_df = dice.generate_counterfactuals(
             data=data,
             continuous_features=data.get_numerical_features(),
             features_to_keep=['gender'],  # Keep gender unchanged
             total_cfs=5
         )

class **DisCount** (ml_model)

   **Bases:** ``CounterFactualExplainer``

   Distributional Counterfactual Explanations with Optimal Transport (DisCount) generator.

   Generates distributional counterfactuals that maintain similar structure to the factual
   distribution while achieving desired prediction outcomes. Uses optimal transport theory
   and Wasserstein distances for robust distributional matching.

   **IMPORTANT: Only compatible with PyTorch models (backend='pytorch').**

   **Parameters:**
      **ml_model** (*Model*) -- Pre-trained model wrapped in Model interface.
      Must be a PyTorch model (``backend='pytorch'``).

   **Raises:**
      **ValueError** -- If ml_model.backend is not 'pytorch'

   **References:**
      You, L., Cao, L., Nilsson, M., Zhao, B., & Lei, L. (2025).
      DIStributional COUNTerfactual Explanation With Optimal Transport.
      In Proceedings of AISTATS 2025 (Oral).
      https://arxiv.org/pdf/2401.13112

   **Mathematical Foundation:**

   DisCount solves a constrained optimization problem:

   .. math::

      \min_x Q = (1-\eta) Q_x(x, \mu) + \eta Q_y(x, \nu)

   subject to:

   .. math::

      \begin{aligned}
      \text{SW}_2(x, x') &\leq U_1 \\
      \text{W}_2(b(x), y^*) &\leq U_2
      \end{aligned}

   where:

   - **x**: Counterfactual distribution (optimized)
   - **x'**: Factual distribution (fixed)
   - **y = b(x)**: Model predictions
   - **y***: Target prediction distribution
   - **SW₂**: Sliced Wasserstein-2 distance
   - **W₂**: Wasserstein-2 distance
   - **U₁, U₂**: Upper bounds on distributional distances

   **Key Features:**

   - Distribution-level counterfactual generation
   - Preserves distributional structure
   - Uses sliced Wasserstein distance for efficient computation
   - Robust to outliers with trimmed distances
   - Statistical guarantees via confidence upper bounds
   - Supports data transformation via COLAData.transform_method parameter
   - Automatically handles inverse transformation of counterfactuals

   **Example:**

   .. code-block:: python

      from xai_cola.ce_generator import DisCount
      from xai_cola.ce_sparsifier.models import Model
      from xai_cola.ce_sparsifier.data import COLAData

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
          lr=5e-2,
          n_proj=10,
          delta=0.15,
          U_1=0.4,
          U_2=0.02,
          max_iter=100
      )

   **generate_counterfactuals** (data=None, factual_class=1, lr=5e-2, n_proj=10, delta=0.15, U_1=0.4, U_2=0.02, l=0.15, r=1, max_iter=100, tau=1e1, silent=False, explain_columns=None)

      Implement the DisCount algorithm to generate counterfactuals.

      **Parameters:**
         * **data** (*COLAData**, **optional*) -- Factual data wrapper
         * **factual_class** (*int**, **default=1*) -- The class of the factual data.
           Normally, we set the factual_class as 1 as the prediction of factual data is 1.
           And we hope the prediction of counterfactual data is 0.
         * **lr** (*float**, **default=5e-2*) -- Learning rate for optimization.
           Controls the step size in gradient descent updates.
         * **n_proj** (*int**, **default=10*) -- Number of random projections for computing sliced Wasserstein distance.
           More projections give better approximation but slower computation.
         * **delta** (*float**, **default=0.15*) -- Trimming constant ∈ (0, 0.5) for robust distance computation.
           Trims delta fraction from both tails of the distribution.
         * **U_1** (*float**, **default=0.4*) -- Upper bound for input distributional distance
           (sliced Wasserstein distance between factual and counterfactual feature distributions).
         * **U_2** (*float**, **default=0.02*) -- Upper bound for output prediction distance
           (Wasserstein distance between counterfactual predictions and target distribution).
         * **l** (*float**, **default=0.15*) -- Lower bound for interval narrowing in the balancing weight η search.
         * **r** (*float**, **default=1*) -- Upper bound for interval narrowing in the balancing weight η search.
         * **max_iter** (*int**, **default=100*) -- Maximum number of optimization iterations.
         * **tau** (*float**, **default=1e1*) -- Step size for manifold gradient descent.
           Cannot be too large or too small.
         * **silent** (*bool**, **default=False*) -- Whether to suppress optimization logs during training.
         * **explain_columns** (*list**, **optional*) -- List of column names that can be modified during optimization.
           If None, all transformed features can be modified.

      **Returns:**
         * **factual_df** (*pd.DataFrame*) -- DataFrame with shape (n_samples, n_features + 1), includes target column
         * **counterfactual_df** (*pd.DataFrame*) -- DataFrame with shape (n_samples, n_features + 1), includes target column.
           Target column values for counterfactual are set based on actual model predictions.

      **Return type:**
         tuple

      **Note:**

      - Only compatible with PyTorch models (backend='pytorch')
      - Supports data transformation via COLAData.transform_method parameter
      - Automatically handles inverse transformation of counterfactuals

      **Example:**

      .. code-block:: python

         # Generate with custom parameters
         factual, cf = explainer.generate_counterfactuals(
             data=data,
             factual_class=1,
             lr=1e-2,        # Smaller learning rate
             n_proj=50,      # More projections for accuracy
             delta=0.1,      # Less trimming
             U_1=0.3,        # Tighter distributional constraint
             U_2=0.01,       # Tighter prediction constraint
             max_iter=200,   # More iterations
             silent=False    # Show progress
         )

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
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier

   # Create and train pipeline
   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('clf', RandomForestClassifier())
   ])
   pipe.fit(X_train, y_train)

   # Setup
   data = COLAData(
       factual_data=df,
       label_column='Risk',
       numerical_features=['Age', 'Income']
   )
   ml_model = Model(model=pipe, backend="sklearn")

   # Create DiCE explainer
   explainer = DiCE(ml_model=ml_model)

   # Generate counterfactuals
   factual, counterfactual = explainer.generate_counterfactuals(
       data=data,
       factual_class=1,
       total_cfs=2,
       continuous_features=['Age', 'Income']
   )

   # Add to data for further processing
   data.add_counterfactuals(counterfactual, with_target_column=True)

With Immutable Features
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Don't change Age or Gender
   factual, cf = explainer.generate_counterfactuals(
       data=data,
       factual_class=1,
       total_cfs=3,
       features_to_keep=['Age', 'Gender'],  # These features stay unchanged
       continuous_features=['Income', 'Duration']
   )

With Feature Ranges
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set realistic bounds for features
   factual, cf = explainer.generate_counterfactuals(
       data=data,
       factual_class=1,
       total_cfs=2,
       continuous_features=['Age', 'Income'],
       permitted_range={
           'Age': [20, 30],                              # Numerical range
           'Education': ['Doctorate', 'Prof-school']     # Categorical values
       }
   )

Using DisCount
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from xai_cola.ce_generator import DisCount
   from xai_cola.ce_sparsifier.data import COLAData
   from xai_cola.ce_sparsifier.models import Model
   import torch.nn as nn

   # Define PyTorch model
   class NeuralNet(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 2)
           )

       def forward(self, x):
           return self.layers(x)

   # Train model
   pytorch_model = NeuralNet(input_dim=10)
   # ... training code ...

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
       U_1=0.3,       # Tighter distributional constraint
       U_2=0.01,      # Tighter prediction constraint
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
   cola = COLA(data=data, ml_model=ml_model)
   cola.set_policy(matcher='ot', attributor='pshap', random_state=42)
   refined = cola.get_refined_counterfactual(limited_actions=5)

   # 5. Visualize
   cola.heatmap_direction(save_path='./results')

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
   cola = COLA(data=data, ml_model=ml_model)
   cola.set_policy(matcher='nn', attributor='pshap', random_state=42)
   refined = cola.get_refined_counterfactual(limited_actions=5)

   # 5. Visualize
   cola.heatmap_binary(save_path='./results')
   cola.stacked_bar_chart(save_path='./results')

Algorithm Comparison
====================

+------------------+------------------+------------------------+
| Aspect           | DiCE             | DisCount               |
+==================+==================+========================+
| Level            | Instance-wise    | Distribution-wise      |
+------------------+------------------+------------------------+
| Backend          | Sklearn          | PyTorch only           |
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

- :doc:`cola` - COLA refinement API
- :doc:`data` - Data interface API
- :doc:`models` - Model interface API
