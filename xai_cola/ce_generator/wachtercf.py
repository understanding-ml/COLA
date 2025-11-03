"""
Wachter Counterfactual Explainer

Implementation based on:
Wachter, S., Mittelstadt, B., & Russell, C. (2017). 
Counterfactual explanations without opening the black box: automated decisions 
and the GDPR. Harv. JL & Tech., 31, 841.

Supports both PyTorch models (with automatic gradients) and sklearn models (with numerical gradients).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .base_explainer import CounterFactualExplainer
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_sparsifier.data import COLAData

SHUFFLE_COUNTERFACTUAL = False


class WachterCF(CounterFactualExplainer):
    """
    Wachter Counterfactual Explainer - A gradient-based counterfactual generator.
    
    Supports both PyTorch models (with automatic gradients) and sklearn models 
    (with numerical gradients).
    """

    def __init__(self, ml_model: Model, data: COLAData = None):
        """
        Initialize WachterCF explainer
        
        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
            Supports 'pytorch' (with automatic gradients) and 'sklearn' (with numerical gradients)
        data : COLAData, optional
            Data wrapper containing factual data
        """
        super().__init__(ml_model, data)
        
        # Check if model is PyTorch (for automatic gradients) or sklearn (for numerical gradients)
        self.use_automatic_grad = (self.ml_model.backend == 'pytorch')
        
        if self.ml_model.backend not in ['pytorch', 'sklearn']:
            raise ValueError(
                f"WachterCF currently supports 'pytorch' and 'sklearn' backends, "
                f"but got backend='{self.ml_model.backend}'. "
                f"For PyTorch, automatic gradients are used. "
                f"For sklearn, numerical gradients are used."
            )

    def generate_counterfactuals(
        self,
        data: COLAData = None,
        factual_class: int = 1,
        features_to_vary: Optional[List[str]] = None,
        target_proba: float = 0.7,
        feature_weights: Optional[List[float]] = None,
        _lambda: float = 10.0,
        optimizer: str = "adam",
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate counterfactuals using Wachter's method.

        Parameters:
        -----------
        data : COLAData, optional
            Factual data wrapper
        factual_class : int, default=1
            The class of the factual data. Counterfactuals will target (1 - factual_class)
        features_to_vary : list, optional
            List of feature names to vary. If None, all features can vary
        target_proba : float, default=0.7
            Target probability for the counterfactual class
        feature_weights : list, optional
            Weights for features in distance computation. If None, equal weights
        _lambda : float, default=10.0
            Weight for the prediction loss term
        optimizer : str, default="adam"
            Optimizer to use: "adam" or "rmsprop"
        lr : float, default=0.01
            Learning rate for optimization
        max_iter : int, default=100
            Maximum number of optimization iterations

        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)
        """
        # Call the data processing logic from the parent class
        self._process_data(data)
        
        x_chosen = self.x_factual_pandas  # factual, dataframe type, without target_column
        
        # Apply transformation if needed
        if self.data.transform_method is not None:
            x_chosen_transformed = self.data._transform(x_chosen)
        else:
            x_chosen_transformed = x_chosen
        
        # Get feature names (transformed if applicable)
        feature_names = x_chosen_transformed.columns.tolist()
        
        # Prepare mask for features to vary
        if features_to_vary is None:
            # All features can vary
            mask = torch.ones(len(feature_names))
        else:
            # Create mask: 1 for features to vary, 0 for features to keep fixed
            mask = torch.zeros(len(feature_names))
            for feat in features_to_vary:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    mask[idx] = 1.0
        
        # Process each instance
        counterfactual_list = []
        for idx in range(len(x_chosen_transformed)):
            query_instance = x_chosen_transformed.iloc[[idx]].values
            query_instance_tensor = torch.FloatTensor(query_instance)
            
            # Generate counterfactual for this instance
            cf_tensor = self._optimize_counterfactual(
                query_instance_tensor,
                mask,
                target_proba=target_proba,
                factual_class=factual_class,
                feature_weights=feature_weights,
                _lambda=_lambda,
                optimizer=optimizer,
                lr=lr,
                max_iter=max_iter,
            )
            
            # Convert back to numpy
            cf_array = cf_tensor.detach().numpy()
            counterfactual_list.append(cf_array[0])
        
        # Create counterfactual DataFrame
        df_counterfactual_transformed = pd.DataFrame(
            counterfactual_list,
            columns=feature_names,
            index=x_chosen_transformed.index
        )
        
        # Inverse transform if needed
        if self.data.transform_method is not None:
            df_counterfactual = self.data._inverse_transform(df_counterfactual_transformed)
        else:
            df_counterfactual = df_counterfactual_transformed
        
        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        # Prepare factual with target column - get directly from COLAData
        factual_df = self.data.get_factual_all()
        
        # Prepare counterfactual with target column
        counterfactual_target_value = 1 - factual_class
        counterfactual_df = df_counterfactual.copy()
        counterfactual_df[self.target_name] = counterfactual_target_value
        
        # Ensure column order matches factual (target column at the end)
        all_columns = factual_df.columns.tolist()
        counterfactual_df = counterfactual_df[all_columns]
        
        return factual_df, counterfactual_df

    def _optimize_counterfactual(
        self,
        query_instance: torch.Tensor,
        mask: torch.Tensor,
        target_proba: float,
        factual_class: int,
        feature_weights: Optional[List[float]] = None,
        _lambda: float = 10.0,
        optimizer: str = "adam",
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> torch.Tensor:
        """
        Optimize counterfactual for a single instance.

        Parameters:
        -----------
        query_instance : torch.Tensor
            Input instance (1, n_features)
        mask : torch.Tensor
            Mask for features to vary (n_features,)
        target_proba : float
            Target probability for counterfactual class
        factual_class : int
            Original class
        feature_weights : list, optional
            Feature weights for distance
        _lambda : float
            Weight for prediction loss
        optimizer : str
            Optimizer name
        lr : float
            Learning rate
        max_iter : int
            Maximum iterations

        Returns:
        --------
        torch.Tensor
            Optimized counterfactual (1, n_features)
        """
        # Initialize counterfactual
        # Use random initialization in [0, 1] range
        cf_initialize = torch.rand(query_instance.shape)
        cf_initialize = cf_initialize * mask.unsqueeze(0) + query_instance * (1 - mask.unsqueeze(0))
        cf_initialize = cf_initialize.clone().detach().requires_grad_(True)

        # Feature weights
        if feature_weights is None:
            feature_weights_tensor = torch.ones(query_instance.shape[1])
        else:
            feature_weights_tensor = torch.FloatTensor(feature_weights)

        # Select optimizer
        if optimizer == "adam":
            optim = torch.optim.Adam([cf_initialize], lr=lr)
        else:
            optim = torch.optim.RMSprop([cf_initialize], lr=lr)

        # Iterative optimization
        for i in range(max_iter):
            optim.zero_grad()
            
            if self.use_automatic_grad:
                # PyTorch model: use automatic gradients
                loss = self._compute_loss_automatic_grad(
                    cf_initialize,
                    query_instance,
                    target_proba,
                    factual_class,
                    feature_weights_tensor,
                    _lambda,
                )
                loss.backward()
                
                # Only update features that are allowed to vary
                with torch.no_grad():
                    cf_initialize.grad = cf_initialize.grad * mask.unsqueeze(0)
            else:
                # sklearn model: use numerical gradients
                grad = self._compute_numerical_gradient(
                    cf_initialize,
                    query_instance,
                    target_proba,
                    factual_class,
                    feature_weights_tensor,
                    _lambda,
                    mask,
                )
                # Manually set gradients
                cf_initialize.grad = grad * mask.unsqueeze(0)
            
            optim.step()

            # Clip values to reasonable range [0, 1] for normalized data
            with torch.no_grad():
                cf_initialize.data = torch.clamp(cf_initialize.data, 0.0, 1.0)

        return cf_initialize.detach()

    def _compute_loss_automatic_grad(
        self,
        cf_initialize: torch.Tensor,
        query_instance: torch.Tensor,
        target_proba: float,
        factual_class: int,
        feature_weights: torch.Tensor,
        _lambda: float,
    ) -> torch.Tensor:
        """
        Compute loss for PyTorch models using automatic gradients.

        Parameters:
        -----------
        cf_initialize : torch.Tensor
            Current counterfactual (1, n_features)
        query_instance : torch.Tensor
            Original instance (1, n_features)
        target_proba : float
            Target probability
        factual_class : int
            Original class
        feature_weights : torch.Tensor
            Feature weights
        _lambda : float
            Weight for prediction loss

        Returns:
        --------
        torch.Tensor
            Loss value
        """
        # Get PyTorch model
        pytorch_model = self.ml_model._get_wrapped_model().model
        
        # Forward pass through model (need gradients)
        output = pytorch_model(cf_initialize)
        # Squeeze to handle batch dimension if needed
        if output.dim() > 0:
            output = output.squeeze()
        
        # For binary classification, apply sigmoid to get probability in [0, 1]
        output_proba = torch.sigmoid(output)
        
        # Get probability of target class (1 - factual_class)
        if factual_class == 1:
            # Target class is 0, so probability of class 0 = 1 - prob(class 1)
            target_class_proba = 1.0 - output_proba
        else:
            # Target class is 1, so probability of class 1 = output_proba
            target_class_proba = output_proba
        
        # Loss 1: Prediction loss (hinge loss for reaching target probability)
        loss1 = F.relu(target_proba - target_class_proba)
        
        # Loss 2: Distance loss (weighted L2 distance)
        diff = cf_initialize - query_instance
        loss2 = torch.sum(feature_weights.unsqueeze(0) * (diff ** 2))
        
        return _lambda * loss1 + loss2

    def _compute_numerical_gradient(
        self,
        cf_initialize: torch.Tensor,
        query_instance: torch.Tensor,
        target_proba: float,
        factual_class: int,
        feature_weights: torch.Tensor,
        _lambda: float,
        mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute numerical gradient for sklearn models using finite differences.

        Parameters:
        -----------
        cf_initialize : torch.Tensor
            Current counterfactual (1, n_features)
        query_instance : torch.Tensor
            Original instance (1, n_features)
        target_proba : float
            Target probability
        factual_class : int
            Original class
        feature_weights : torch.Tensor
            Feature weights
        _lambda : float
            Weight for prediction loss
        mask : torch.Tensor
            Mask for features to vary
        eps : float
            Step size for finite differences

        Returns:
        --------
        torch.Tensor
            Numerical gradient (1, n_features)
        """
        cf_np = cf_initialize.detach().clone().numpy()
        n_features = cf_np.shape[1]
        gradient = np.zeros_like(cf_np)
        
        # Compute loss at current point
        current_loss = self._compute_loss_sklearn(
            cf_np,
            query_instance.numpy(),
            target_proba,
            factual_class,
            feature_weights.numpy(),
            _lambda,
        )
        
        # Compute gradient using finite differences
        for j in range(n_features):
            if mask[j].item() == 0:
                # Feature is fixed, gradient is 0
                continue
                
            # Perturb feature j
            cf_perturbed = cf_np.copy()
            cf_perturbed[0, j] += eps
            
            # Compute loss at perturbed point
            perturbed_loss = self._compute_loss_sklearn(
                cf_perturbed,
                query_instance.numpy(),
                target_proba,
                factual_class,
                feature_weights.numpy(),
                _lambda,
            )
            
            # Finite difference gradient
            gradient[0, j] = (perturbed_loss - current_loss) / eps
        
        return torch.FloatTensor(gradient)

    def _compute_loss_sklearn(
        self,
        cf_instance: np.ndarray,
        query_instance: np.ndarray,
        target_proba: float,
        factual_class: int,
        feature_weights: np.ndarray,
        _lambda: float,
    ) -> float:
        """
        Compute loss for sklearn models (numpy-based).

        Parameters:
        -----------
        cf_instance : np.ndarray
            Current counterfactual (1, n_features)
        query_instance : np.ndarray
            Original instance (1, n_features)
        target_proba : float
            Target probability
        factual_class : int
            Original class
        feature_weights : np.ndarray
            Feature weights
        _lambda : float
            Weight for prediction loss

        Returns:
        --------
        float
            Loss value
        """
        # Get prediction probability
        proba = self.ml_model.predict_proba(cf_instance)
        
        # Get probability of target class (1 - factual_class)
        if factual_class == 1:
            # Target class is 0
            target_class_proba = 1.0 - proba[0]
        else:
            # Target class is 1
            target_class_proba = proba[0]
        
        # Loss 1: Prediction loss (hinge loss)
        loss1 = max(0.0, target_proba - target_class_proba)
        
        # Loss 2: Distance loss (weighted L2 distance)
        diff = cf_instance - query_instance
        loss2 = np.sum(feature_weights * (diff ** 2))
        
        return _lambda * loss1 + loss2
