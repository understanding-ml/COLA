==============
Models API
==============

.. currentmodule:: xai_cola.ce_sparsifier.models

Module contents
===============

class **Model** (model, backend, preprocessor=None)

   **Bases:** ``object``

   Class wrapper for pre-trained models with optional preprocessing pipeline.

   Provides a unified interface for machine learning models from different frameworks
   (scikit-learn and PyTorch) with consistent ``predict()`` and ``predict_proba()`` methods.

   **Parameters:**
      * **model** (*object*) -- Pre-trained model (sklearn or PyTorch model).
        Can also be a sklearn Pipeline object.
      * **backend** (*str*) -- Model framework identifier.
        Options: ``"sklearn"`` or ``"pytorch"``.
      * **preprocessor** (*sklearn.compose.ColumnTransformer or sklearn.pipeline.Pipeline**, **optional*) --
        Preprocessing pipeline to apply before model prediction.
        If model is already a Pipeline, this parameter will be ignored.
        Default is None.

   **Raises:**
      **ValueError** -- If backend is not one of the supported frameworks.

   **Note:**

   For sklearn models, if your model is a Pipeline, the Model class will automatically
   detect it and handle preprocessing internally. The preprocessor parameter is only
   needed when you have a separate preprocessor and classifier.

   **Example:**

   .. code-block:: python

      from xai_cola.ce_sparsifier.models import Model
      from sklearn.ensemble import GradientBoostingClassifier
      from sklearn.preprocessing import StandardScaler
      from sklearn.pipeline import Pipeline

      # Option 1: Model with separate preprocessor
      scaler = StandardScaler()
      scaler.fit(X_train)
      clf = GradientBoostingClassifier()
      clf.fit(scaler.transform(X_train), y_train)

      model = Model(model=clf, backend="sklearn", preprocessor=scaler)

      # Option 2: Model with Pipeline (recommended)
      pipe = Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())])
      pipe.fit(X_train, y_train)

      model = Model(model=pipe, backend="sklearn")

      # Option 3: PyTorch model
      import torch.nn as nn

      class NeuralNet(nn.Module):
          def __init__(self):
              super().__init__()
              self.layers = nn.Sequential(
                  nn.Linear(10, 64),
                  nn.ReLU(),
                  nn.Linear(64, 2)
              )

          def forward(self, x):
              return self.layers(x)

      torch_model = NeuralNet()
      # ... train model ...

      model = Model(model=torch_model, backend="pytorch")

   **predict** (x_factual)

      Generate predictions using the pre-trained model.

      **Parameters:**
         **x_factual** (*np.ndarray or pd.DataFrame*) -- Input data for prediction (raw data)

      **Returns:**
         Predictions

      **Return type:**
         np.ndarray

      **Example:**

      .. code-block:: python

         model = Model(model=pipe, backend="sklearn")
         predictions = model.predict(X_test)

   **predict_proba** (X)

      Predict probability function that returns the probability distribution for each class.

      **Parameters:**
         **X** (*np.ndarray or pd.DataFrame*) -- Input data for which to predict probabilities (raw data)

      **Returns:**
         Probability of positive class (class 1) for binary classification

      **Return type:**
         np.ndarray

      **Example:**

      .. code-block:: python

         model = Model(model=pipe, backend="sklearn")
         probabilities = model.predict_proba(X_test)

   **to** (device)

      Move PyTorch model to specified device.

      **Parameters:**
         **device** (*str or torch.device*) -- Device to move model to (e.g., 'cuda', 'cpu')

      **Returns:**
         Returns self for method chaining

      **Return type:**
         Model

      **Raises:**
         **NotImplementedError** -- If backend is not 'pytorch'

      **Example:**

      .. code-block:: python

         model = Model(model=torch_model, backend="pytorch")
         model.to("cuda")  # Move to GPU
         predictions = model.predict(X_test)

   **__call__** (x)

      Call the model directly (mainly for PyTorch models).

      **Parameters:**
         **x** (*torch.Tensor or np.ndarray*) -- Input data

      **Returns:**
         Model output

      **Return type:**
         torch.Tensor or np.ndarray

      **Raises:**
         **NotImplementedError** -- If backend is not 'pytorch'

      **Example:**

      .. code-block:: python

         model = Model(model=torch_model, backend="pytorch")
         import torch
         x = torch.randn(32, 10)
         output = model(x)  # Direct call

Supported Backends
==================

COLA supports the following machine learning frameworks:

Scikit-learn
------------

**Backend string:** ``"sklearn"``

**Requirements:**

- scikit-learn installed
- Model must have ``predict()`` and ``predict_proba()`` methods

**Supported Model Types:**

- All scikit-learn classifiers (RandomForest, GradientBoosting, LogisticRegression, etc.)
- Scikit-learn Pipeline objects (automatically detected and handled)
- Any custom classifier implementing the scikit-learn interface

**Key Features:**

- Automatic Pipeline detection
- Handles both raw features and preprocessed data
- Supports separate preprocessor parameter for non-Pipeline models

PyTorch
-------

**Backend string:** ``"pytorch"``

**Requirements:**

- PyTorch installed
- Model must be a ``torch.nn.Module``
- Model should output logits or probabilities for classification

**Key Features:**

- GPU support via ``.to(device)`` method
- Automatic tensor conversion from numpy arrays/pandas DataFrames
- Direct model call support via ``__call__()`` method
- Handles NaN values in predictions

**Note:**

For PyTorch models, the Model class automatically:

- Converts input to torch.FloatTensor
- Sets model to evaluation mode
- Handles device placement (CPU/GPU)
- Applies softmax for probability predictions

Examples
========

Basic Usage with sklearn Pipeline
----------------------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from xai_cola.ce_sparsifier.models import Model

   # Create and train pipeline
   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('clf', RandomForestClassifier())
   ])
   pipe.fit(X_train, y_train)

   # Wrap for COLA
   ml_model = Model(model=pipe, backend="sklearn")

   # Use with COLA
   from xai_cola import COLA
   from xai_cola.ce_sparsifier.data import COLAData

   data = COLAData(factual_data=df, label_column='Risk')
   data.add_counterfactuals(cf_df)

   cola = COLA(data=data, ml_model=ml_model)

sklearn with Separate Preprocessor
-----------------------------------

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from xai_cola.ce_sparsifier.models import Model

   # Fit preprocessor
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)

   # Train classifier on scaled data
   clf = LogisticRegression()
   clf.fit(X_train_scaled, y_train)

   # Wrap both
   ml_model = Model(
       model=clf,
       backend="sklearn",
       preprocessor=scaler
   )

   # Automatically scales input before prediction
   predictions = ml_model.predict(X_test)  # X_test is raw data

sklearn with ColumnTransformer
-------------------------------

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, OrdinalEncoder
   from sklearn.ensemble import GradientBoostingClassifier
   from xai_cola.ce_sparsifier.models import Model

   # Define custom preprocessor
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), ['Age', 'Income']),
       ('cat', OrdinalEncoder(), ['Gender', 'Education'])
   ])
   preprocessor.fit(X_train)

   # Train classifier on transformed data
   X_train_transformed = preprocessor.transform(X_train)
   clf = GradientBoostingClassifier()
   clf.fit(X_train_transformed, y_train)

   # Wrap both
   ml_model = Model(
       model=clf,
       backend="sklearn",
       preprocessor=preprocessor
   )

   # Use with raw data
   predictions = ml_model.predict(X_test)

PyTorch Neural Network (CPU)
-----------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from xai_cola.ce_sparsifier.models import Model

   class NeuralNet(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(input_dim, 64),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Linear(32, 2)
           )

       def forward(self, x):
           return self.layers(x)

   # Train model
   model = NeuralNet(input_dim=10)
   # ... training code ...

   # Wrap for COLA
   ml_model = Model(model=model, backend="pytorch")

   # Predictions work with numpy arrays or DataFrames
   predictions = ml_model.predict(X_test)
   probabilities = ml_model.predict_proba(X_test)

PyTorch Neural Network (GPU)
-----------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from xai_cola.ce_sparsifier.models import Model

   # Define and train model
   model = NeuralNet(input_dim=10)
   # ... training code on GPU ...

   # Wrap for COLA
   ml_model = Model(model=model, backend="pytorch")

   # Move to GPU
   ml_model.to("cuda")

   # Predictions automatically handle device placement
   predictions = ml_model.predict(X_test)

   # Or use directly with tensors
   x_tensor = torch.randn(32, 10).to("cuda")
   output = ml_model(x_tensor)

PyTorch with Preprocessing
---------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from sklearn.preprocessing import StandardScaler
   from xai_cola.ce_sparsifier.models import Model

   # Train preprocessor
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)

   # Train PyTorch model on scaled data
   model = NeuralNet(input_dim=X_train_scaled.shape[1])
   # ... training code using X_train_scaled ...

   # Wrap both
   ml_model = Model(
       model=model,
       backend="pytorch",
       preprocessor=scaler
   )

   # Automatically scales input before prediction
   predictions = ml_model.predict(X_test)  # X_test is raw data

Comparing Different Model Types
--------------------------------

.. code-block:: python

   from xai_cola import COLA
   from xai_cola.ce_sparsifier.data import COLAData
   from xai_cola.ce_sparsifier.models import Model

   # Prepare data
   data = COLAData(factual_data=df, label_column='Risk')
   data.add_counterfactuals(cf_df)

   # Scikit-learn model
   sklearn_model = Model(model=sklearn_pipe, backend="sklearn")
   cola_sklearn = COLA(data=data, ml_model=sklearn_model)
   sklearn_results = cola_sklearn.get_refined_counterfactual(limited_actions=10)

   # PyTorch model
   pytorch_model = Model(model=torch_model, backend="pytorch")
   cola_pytorch = COLA(data=data, ml_model=pytorch_model)
   pytorch_results = cola_pytorch.get_refined_counterfactual(limited_actions=10)

   # Both work seamlessly with COLA!

See Also
========

- :doc:`cola` - COLA main class
- :doc:`data` - Data interface documentation
